# Linear Regression

## Layman's Explanation

Linear regression is a method used to find the relationship between two variables by fitting a straight line to the observed data. Imagine you have a scatter plot of data points, and you want to draw a straight line that best represents the overall trend. This line can then be used to predict new values.

## Technical Explanation

Linear regression is a statistical method to model the relationship between a dependent variable and one or more independent variables. The goal is to find the coefficients that minimize the sum of squared differences between the actual and predicted values.

### Production-Level Implementation

The production-level implementation using `scikit-learn` includes:
- **Data Validation**: Ensures inputs are numpy arrays with compatible shapes.
- **Error Handling**: Manages common issues, such as using the model before training.
- **R-squared Score**: An additional method (`score`) to evaluate the model using the R-squared metric.
- **Data Loading**: Loads data from a CSV file for easier preprocessing.

### Usage

#### Loading Data

Data should be placed in the `data` directory inside the `linear_regression` folder. Here's how to load and use it:

```python
from linear_regression import LinearRegression
import numpy as np
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_data.csv')
model = LinearRegression()
X, y = model.load_data(data_path)

#Training the Model
# Initialize and train the model
model.fit(X, y)

#Making Predictions
predictions = model.predict(X)
print("Predictions:", predictions)

#Evaluating the Model
r2_score_value = model.score(X, y)
print("R-squared score:", r2_score_value)

#Inference
#To make predictions on new data:
new_data = np.array([[6], [7]])
inference_predictions = model.predict(new_data)
print("Inference predictions for new data:", inference_predictions)



