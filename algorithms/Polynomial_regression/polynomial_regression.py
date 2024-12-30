```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
import os

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = SklearnLinearRegression()

    def fit(self, X, y):
        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal")

        # Transform features to polynomial features
        X_poly = self.poly_features.fit_transform(X)

        # Fit the model
        self.model.fit(X_poly, y)

    def predict(self, X):
        # Check if the model is trained
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model is not trained yet. Call `fit` before `predict`.")

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        # Transform features to polynomial features
        X_poly = self.poly_features.transform(X)

        return self.model.predict(X_poly)

    def score(self, X, y):
        # Validate inputs
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")

        # Make predictions
        predictions = self.predict(X)

        # Calculate the R-squared score
        return r2_score(y, predictions)

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
    model = PolynomialRegression(degree=3)
    X, y = model.load_data(data_path)

    # Train the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)

    # Calculate and print R-squared score
    r2_score_value = model.score(X, y)
    print("R-squared score:", r2_score_value)

    # Inference on new data
    new_data = np.array([[6], [7]])
    inference_predictions = model.predict(new_data)
    print("Inference predictions for new data:", inference_predictions)
