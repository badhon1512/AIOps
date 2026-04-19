#Ref: https://mlflow.org/docs/latest/ml/getting-started/quickstart/
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}


#Enable autologging for scikit-learn
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.sklearn.autolog()

mlflow.set_experiment("Train-Model-Experiment")

with mlflow.start_run(run_name="Train-Model-Run", run_id="3ff352f92cd64e60ad3a9edc52cd47a4") as run:
    #mlflow.log_params(params)
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    #mlflow.sklearn.log_model(lr, "logistic_regression_model")