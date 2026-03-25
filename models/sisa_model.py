import numpy as np
import pandas as pd
import copy
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

NUM_SHARDS = 5
NUM_SLICES = 5
RANDOM_STATE = 42

class OptimizedSISA:
    def __init__(self, n_shards, n_slices):
        self.n_shards = n_shards
        self.n_slices = n_slices
        self.shards = []
        self.slices = []
        self.checkpoints = {}
        self.final_models = {}

    def _new_model(self):
        return LogisticRegression(max_iter=1000, solver="lbfgs")

    def fit(self, X, y):
        n = len(X)
        shard_size = n // self.n_shards
        indices_list = np.arange(n)

        for s in range(self.n_shards):
            start = s * shard_size
            end = (s + 1) * shard_size if s < self.n_shards - 1 else n
            shard_idx = indices_list[start:end]
            self.shards.append(shard_idx)

            slice_size = len(shard_idx) // self.n_slices
            shard_slices = []

            for k in range(self.n_slices):
                ss = k * slice_size
                se = (k + 1) * slice_size if k < self.n_slices - 1 else len(shard_idx)
                shard_slices.append(shard_idx[ss:se])

            self.slices.append(shard_slices)
            self.checkpoints[s] = {}

            model = self._new_model()
            cumulative = np.array([], dtype=int)

            for k in range(self.n_slices):
                cumulative = np.concatenate([cumulative, shard_slices[k]])
                model.fit(X[cumulative], y[cumulative])
                self.checkpoints[s][k] = copy.deepcopy(model)

            self.final_models[s] = self.checkpoints[s][self.n_slices - 1]

    def predict_proba(self, X):
        probs = []
        for s in range(self.n_shards):
            probs.append(self.final_models[s].predict_proba(X)[:, 1])
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def unlearn(self, X, y, forget_indices):
        start = time.time()
        retrained = 0

        for s in range(self.n_shards):
            shard_idx = self.shards[s]
            shard_slices = self.slices[s]

            affected = np.intersect1d(shard_idx, forget_indices)
            if len(affected) == 0:
                continue

            retrained += 1

            first_bad = 0
            for k in range(self.n_slices):
                if np.intersect1d(shard_slices[k], forget_indices).size > 0:
                    first_bad = k
                    break

            if first_bad == 0:
                model = self._new_model()
                cumulative = np.array([], dtype=int)
            else:
                model = copy.deepcopy(self.checkpoints[s][first_bad - 1])
                cumulative = np.concatenate(shard_slices[:first_bad])

            for k in range(first_bad, self.n_slices):
                clean_slice = np.setdiff1d(shard_slices[k], forget_indices)
                cumulative = np.concatenate([cumulative, clean_slice])
                model.fit(X[cumulative], y[cumulative])
                self.checkpoints[s][k] = copy.deepcopy(model)

            self.final_models[s] = self.checkpoints[s][self.n_slices - 1]

        print(f"SISA Unlearning complete. Shards retrained: {retrained}/{self.n_shards}")

    def learn_new_data(self, X, y, new_indices):
        """
        Incrementally learns new data by appending it to the last shard.
        - X, y: Full dataset (including new data).
        - new_indices: Indices of the new data in X, y.
        """
        # Append to the last shard
        last_shard_idx = self.n_shards - 1
        
        # Update Shard Indices
        current_shard_indices = self.shards[last_shard_idx]
        updated_shard_indices = np.concatenate([current_shard_indices, new_indices])
        self.shards[last_shard_idx] = updated_shard_indices
        
        # Update Last Slice Indices
        last_slice_idx = self.n_slices - 1
        current_slice_indices = self.slices[last_shard_idx][last_slice_idx]
        updated_slice_indices = np.concatenate([current_slice_indices, new_indices])
        self.slices[last_shard_idx][last_slice_idx] = updated_slice_indices
        
        # Retrain the Last Shard (Last Slice Logic)
        # We need to rebuild the model for this shard to include the new data
        # We can start from the second-to-last slice checkpoint to save time
        
        # Get base model (trained on slices 0 to N-2)
        if self.n_slices > 1:
            base_model = copy.deepcopy(self.checkpoints[last_shard_idx][self.n_slices - 2])
            
            # Construct cumulative data for the last slice (Base + Last Slice)
            # Actually, base_model is already trained on Slices 0..N-2.
            # We just need to partial_fit or fit on the NEW Cumulative? 
            # sklearn LogisticRegression doesn't support online partial_fit easily for 'lbfgs'.
            # So we must re-fit on the full cumulative data for this shard.
            
            # Gather all indices for this shard
            all_shard_indices = self.shards[last_shard_idx]
            
            # To respect the slicing architecture (though for last slice it implies full shard):
            model = self._new_model()
            model.fit(X[all_shard_indices], y[all_shard_indices])
            
            # Update Checkpoint and Final Model
            self.checkpoints[last_shard_idx][last_slice_idx] = copy.deepcopy(model)
            self.final_models[last_shard_idx] = model
        else:
            # Single slice per shard? Just retrain shard.
            all_shard_indices = self.shards[last_shard_idx]
            model = self._new_model()
            model.fit(X[all_shard_indices], y[all_shard_indices])
            self.final_models[last_shard_idx] = model

        print(f"SISA Incremental Learning complete. Shard {last_shard_idx} updated with {len(new_indices)} new records.")
