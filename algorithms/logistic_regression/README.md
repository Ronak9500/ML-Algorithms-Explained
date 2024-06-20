# Logistic Regression

## Layman's Explanation

Logistic regression is a method used for predicting binary outcomes. It helps determine whether an event will happen or not based on one or more predictor variables. Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities of the outcome.

### Example:

Imagine you want to predict whether a student will pass or fail a class based on the number of hours they study. Logistic regression will help you determine the probability of passing or failing given the study hours.

## Technical Explanation

Logistic regression works by applying a logistic function (also called the sigmoid function) to the linear combination of input features. This function transforms the output to a value between 0 and 1, representing the probability of the positive class.

### Production-Level Implementation

The production-level implementation using `scikit-learn` includes:
- **Data Validation**: Ensures inputs are numpy arrays with compatible shapes.
- **Error Handling**: Manages common issues, such as using the model before training.
- **Evaluation Metrics**: Provides methods to calculate accuracy, precision, recall, and F1 score.
- **Data Loading**: Loads data from a CSV file for easier preprocessing.

### Usage

#### Loading Data

Data should be placed in the `data` directory inside the `logistic_regression` folder. Here's how to load and use it:

```python
from logistic_regression import LogisticRegression
import numpy as np
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), 'data', 'logistic_sample_data.csv')
model = LogisticRegression()
X, y = model.load_data(data_path)

#Training the Model
# Initialize and train the model
model.fit(X, y)

#Making Predictions
predictions = model.predict(X)
print("Predictions:", predictions)

#Evaluating the Model
scores = model.score(X, y)
print("Evaluation Metrics:", scores)

#Inference
#To make predictions on new data:
new_data = np.array([[6], [7]])
inference_predictions = model.predict(new_data)
inference_probabilities = model.predict_proba(new_data)
print("Inference predictions for new data:", inference_predictions)
print("Inference probabilities for new data:", inference_probabilities)
