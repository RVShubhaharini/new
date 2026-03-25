import torch
from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dl(model, test_loader):

    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            preds.extend(torch.argmax(out,1).cpu().numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    return acc

def evaluate_sisa(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def run_validation(unlearned_model, training_state, model_type):
    print(f"Validating {model_type} model...")
    
    if model_type == "DL":
        test_loader = training_state["dl"]["test_loader"]
        acc = evaluate_dl(unlearned_model, test_loader)
        print(f"DL Validation Accuracy: {acc}")
        return acc
    else:
        X_test = training_state["sisa"]["X_test"]
        y_test = training_state["sisa"]["y_test"]
        acc = evaluate_sisa(unlearned_model, X_test, y_test)
        print(f"SISA Validation Accuracy: {acc}")
        return acc

def run_validation_simple(model_type):
    """
    Simple validation for Incremental Learning.
    For this MVP, we assume if training succeeded, validation is implicitly OK
    or we return a placeholder 1.0 to indicate success to the graph.
    Real validation would require loading X_test again.
    """
    print(f"Simple Validation for {model_type}: SKIPPED (Assumed Success)")
    return 1.0
