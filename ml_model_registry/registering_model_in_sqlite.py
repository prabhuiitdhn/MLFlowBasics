import mlflow
from mlflow.models import infer_signature
import numpy as np
from sklearn import datasets
import pickle
# load the model into memory
loaded_model = pickle.load(open(filename, "rb"))

# create a signature for the model based on the input and output data
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
signature = infer_signature(diabetes_X, diabetes_y)

# log and register the model using MLflow scikit-learn API
mlflow.set_tracking_uri("sqlite:///mlruns.db")
reg_model_name = "SklearnLinearRegression"
print("--")
mlflow.sklearn.log_model(
    loaded_model,
    "sk_learn",
    serialization_format="cloudpickle",
    signature=signature,
    registered_model_name=reg_model_name,
)
