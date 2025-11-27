
from src.model_utils import train_model, load_model

def test_training():
    model = train_model()
    assert model is not None

def test_model_file_saved():
    train_model()
    import os
    assert os.path.exists("model.pkl")

def test_predict_method():
    model = train_model()
    assert hasattr(model, "predict")
    assert model.predict([[2]])[0] == 0
    assert model.predict([[3]])[0] == 1
