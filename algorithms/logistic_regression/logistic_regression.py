import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

class LogisticRegression:
    def __init__(self):
        self.model = SklearnLogisticRegression()

    def fit(self, X, y):
        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal")

        # Fit the model
        self.model.fit(X, y)

    def predict(self, X):
        # Check if the model is trained
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model is not trained yet. Call `fit` before `predict`.")

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        return self.model.predict(X)

    def predict_proba(self, X):
        # Check if the model is trained
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model is not trained yet. Call `fit` before `predict`.")

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        return self.model.predict_proba(X)

    def score(self, X, y):
        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        # Make predictions
        predictions = self.predict(X)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def load_data(self, file_path):
        # Load data from a CSV file
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y

# Example usage
if __name__ == "__main__":
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv')
    model = LogisticRegression()
    X, y = model.load_data(data_path)

    # Train the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

    # Calculate and print evaluation metrics
    scores = model.score(X, y)
    print("Evaluation Metrics:", scores)

    # Inference on new data
    new_data = np.array([[1.5], [3.5]])
    inference_predictions = model.predict(new_data)
    inference_probabilities = model.predict_proba(new_data)
    print("Inference predictions for new data:", inference_predictions)
    print("Inference probabilities for new data:", inference_probabilities)
