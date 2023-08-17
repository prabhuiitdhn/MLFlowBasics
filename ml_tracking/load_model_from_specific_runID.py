import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("file:///C:/Users/uib43225/PycharmProjects/DSAlgo/CodingPractice/MLflow/ml_tracking/mlruns/0/9dd654a1b2aa4cd8aa5be32ebd901108/artifacts/model")


predictions = model.predict(X_test)
print(predictions)
