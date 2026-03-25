import joblib
import torch
from models.dl_unlearning_model import BankNet

def predict_ml(X):
    model = joblib.load("models/sisa_model_baseline.pkl")
    return model.predict(X)

def predict_dl(X):
    model = BankNet(X.shape[1])
    model.load_state_dict(torch.load("models/dl_model.pth"))
    model.eval()
    return torch.argmax(model(X), dim=1)
