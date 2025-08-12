import os
from src.train_rf_model import train_and_save_model

def test_training_produces_model():
    model_path = "outputs/test_model.joblib"
    train_and_save_model(model_path)
    assert os.path.exists(model_path), "Model file not created"
