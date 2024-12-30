# Polynomial Regression

## Layman's Explanation
Polynomial regression is an extension of linear regression where the relationship between the independent variable \(X\) and the dependent variable \(y\) is modeled as an \(n\)th degree polynomial. Imagine you have a scatter plot of data points, and instead of fitting a straight line, you fit a curved line that better captures the trend of the data.

## Technical Explanation
Polynomial regression is a form of regression analysis in which the relationship between the dependent variable and the independent variable is modeled as an \(n\)th degree polynomial. The goal is to find the coefficients that minimize the sum of squared differences between the actual and predicted values.

## Production-Level Implementation
The production-level implementation using scikit-learn includes:

- **Data Validation**: Ensures inputs are numpy arrays with compatible shapes.
- **Error Handling**: Manages common issues, such as using the model before training.
- **R-squared Score**: An additional method (score) to evaluate the model using the R-squared metric.
- **Data Loading**: Loads data from a CSV file for easier preprocessing.

## Usage

### Loading Data
Data should be placed in the `data` directory inside the `polynomial_regression` folder. Here's how to load and use it:

```python
from polynomial_regression import PolynomialRegression
import numpy as np
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv')
model = PolynomialRegression(degree=3)
X, y = model.load_data(data_path)

#Training the Model
#Initialize and train the model:
model.fit(X, y)

#Making Predictions
#Make predictions on the training data:
predictions = model.predict(X)
print("Predictions:", predictions)

#Evaluating the Model
#Evaluate the model using the R-squared score:
r2_score_value = model.score(X, y)
print("R-squared score:", r2_score_value)

#Inference
#To make predictions on new data:
new_data = np.array([[6], [7]])
inference_predictions = model.predict(new_data)
print("Inference predictions for new data:", inference_predictions)

