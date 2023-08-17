import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fetching an MLflow Model from the Model Registry

# model_name = "sk-learn-random-forest-reg-model"
# model_version = 1
#
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
#
# print(model.predict(X_test))

# Fetch the latest model version in a specific stage
import mlflow.pyfunc

model_name = "sk-learn-random-forest-reg-model"
stage = "production"
# NEED TO REGISTER MODEL IN PRODUCTION

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
#
# # model.predict(X_test)
# print(model.predict(X_test))

from mlflow import MlflowClient

client = MlflowClient()
client.update_model_version(
    name="sk-learn-random-forest-reg-model",
    version=str(1),
    description="This model version is a scikit-learn random forest containing 100 decision trees",
)
