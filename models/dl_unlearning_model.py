# ============================================================
# DL UNLEARNING MODULE FOR AGENTIC BANKING SYSTEM
# SSD + Targeted Gradient Ascent + Light Recovery
# (LOGIC UNCHANGED – SYSTEM READY)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL DEFINITION
# ============================================================

class BankNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# TRAIN BASELINE MODEL
# ============================================================

def train_dl(data_path="data/bank.csv", fit_model=True):

    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["deposit"], random_state=42
    )

    X_train = train_df.drop(columns=["deposit"])
    if "customer_id" in X_train.columns:
        X_train = X_train.drop(columns=["customer_id"])
        
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    num_cols = X_train.select_dtypes(include=["number"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_cols])

    X_train_enc = np.hstack([
        X_train[num_cols].values,
        encoder.transform(X_train[cat_cols])
    ])

    scaler = StandardScaler()
    scaler.fit(X_train_enc)

    def process(df_data):
        y = df_data["deposit"].map({"yes":1,"no":0}).values
        X_enc = np.hstack([
            df_data[num_cols].values,
            encoder.transform(df_data[cat_cols])
        ])
        X_scaled = scaler.transform(X_enc)
        return torch.FloatTensor(X_scaled), torch.LongTensor(y)

    X_train_ts, y_train_ts = process(train_df)
    X_test_ts, y_test_ts = process(test_df)

    train_loader = DataLoader(TensorDataset(X_train_ts, y_train_ts), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_ts, y_test_ts), batch_size=64)

    model = BankNet(X_train_ts.shape[1]).to(device)
    
    if fit_model:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for _ in range(15):
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x.to(device)), y.to(device))
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), "models/dl_model_baseline.pth")

    return model, train_df, test_loader, process, encoder, scaler, num_cols, cat_cols, len(df)
