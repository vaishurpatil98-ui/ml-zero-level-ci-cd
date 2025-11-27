import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_model():
    # Training data: numbers and labels (0=even, 1=odd)
    X = np.array([[i] for i in range(100)])
    y = np.array([i % 2 for i in range(100)])

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    return model

def load_model():
    return joblib.load("model.pkl")
