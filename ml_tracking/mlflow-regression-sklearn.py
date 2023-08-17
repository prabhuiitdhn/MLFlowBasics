import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


# mlflow.set_tracking_uri("http://localhost:5000") # keeps track of it in the server.
# mlflow.set_tracking_uri("file:/C:/Users/uib43225/PycharmProjects/DSAlgo/CodingPractice/MLflow/ml_tracking/test_tracking")
# tracking_uri = mlflow.get_tracking_uri()
# print("Current tracking uri: {}".format(tracking_uri))

mlflow.autolog() # this will use for logging all the neccessary parameters/artifacts.
                              # If we want to disable the autolog and start individual parameter/artifacts to save then
                              # we can disable it.
# mlflow.log_metric("RMSE", 250.)
# mlflow.log_artifacts("./data")
# mlflow.log_params(params = {"learning_rate": 0.01, "n_estimators": 10})
# mlflow.log_param("learning_rate", 0.01)
# mlflow.log_dict()
# mlflow.log_figure()
# mlflow.log_text()

# mlflow.sklearn.log_model() # this will log the sklearn flavor of model
# mlflow.sklearn.load_model() # this can load the sklearn flavored model using runid


db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=4)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
