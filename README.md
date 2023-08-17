MLFLOW: An open source platform for the machine learning lifecycle.

https://mlflow.org/docs/latest/quickstart.html

Create a new envioronment with python 3.8 (recommended.)

      conda create --name MLFlowPackages python=3.8
      
      conda activate MLFlowPackages
      
      pip install mlflow

MLFLow has 5 individual module Can be used for:
  1. MLflow Tracking: Tracking ML experiments to record and compare model parameters, evaluate performance, and manage artifacts
  2. MLflow Projects:Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production 
  3. MLflow Models:Packaging and deploying models from a variety of ML libraries to a variety of model serving and inference platforms 
  4. MLflow Registry:Collaboratively managing a central model store, including model versioning, stage transitions, and annotations
  5. MLflow Recipes:Accelerating iterative development with templates and reusable scripts for a variety of common modeling tasks


# MLFLOW Tracking.
    mlflow.autolog(): It is used for logging all the params/artifacts used for training the model
    mlflow ui : It helps to visualise the artifacts and all logged for an experiment in the MLflow ui

MLflow stores tracking data and artifacts in an mlruns/subdirectory

We can track your runs with a tracking server, on shred filesystem
    
    mlflow.set_tracking_uri("http://192.168.0.1:5000")
    
    export MLFLOW_TRACKING_URI=http://192.168.0.1:5000
    
    mlflow.set_tracking_uri("file:/C:/Users/uib43225/PycharmProjects/DSAlgo/CodingPractice/MLflow/ml_tracking/test_tracking") # For local path
    
    mlflow server// used for opening mlflow ui
    
    mlflow run //used for running the MLProject file which contains binding of project with entry points.

MLflow tracking is organised in concepts of run, which stores the following information

1. code version: hash map
2. Start & End Time: in meta.yaml file
3. Source: inputs file
4. Parameters: params folder
5. Metrics: metrics folder
6. Artifacts: artifacts folder (model[conda.yaml, MLModel, model.pck, pythin_env.yaml, requirement.txt])

MLflow can be recorded to local filesor SQLAlchemy-compatibel databses, or tracking server.
To log runs remotely, set the MLFLOW_TRACKING_URI environment variable to a tracking server’s URI or call mlflow.set_tracking_uri().

    # THIS IS HELPFUL IF WE WANT TO STORE OR CONNECT THROUGH MYSQL
    https://mlflow.org/docs/latest/tracking.html
    mlflow server --backend-store-uri=mysql+pymysql://root:Conti.1234@localhost:3306/prabhu_test_db ; It helps to store the mlflow experiments which store in backend-store in sql but artifacts not in sql
    
    THis can help to store backend in backend_store_uri and artifacts in s3
    mlflow server \
      --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb \
      --artifacts-destination s3://bucket_name \
      --host remote_host


# MLflow projects

It should have MLProject [a yaml file which contains name, environment and entry points.]
also, It should have mentioned environment. which can be run using
mlflow run localrepo/github repo -P parameters.
used for delpolying the model locally:

    mlflow models serve -m C:\Users\uib43225\PycharmProjects\DSAlgo\CodingPractice\MLflow\ml_projects\mlruns\0\0662b09809ad4b0c88f2252958b52db9\artifacts\model -p 1234
    if we don't want any env then we can use
     --env-manager local
     --port 8000
    
     mlflow models serve -m C:\Users\uib43225\PycharmProjects\DSAlgo\CodingPractice\MLflow\ml_projects\mlruns\0\0662b09809ad4b0c88f2252958b52db9\artifacts\model -port 8000 --env-manager local
    
    mlflow models build-docker -m C:\Users\uib43225\PycharmProjects\DSAlgo\CodingPractice\MLflow\ml_projects\mlruns\0\0662b09809ad4b0c88f2252958b52db9\artifacts\model -n my-docker-image --enable-mlserver
    mlflow models build-docker -m C:\Users\uib43225\PycharmProjects\DSAlgo\PracticeMLOps\MLflow\ml_projects\mlruns\0\0662b09809ad4b0c88f2252958b52db9\artifacts\model -n model-in-docker-image --enable-mlserver
    [for building docker, the docker setup or docker is needed to be installed in system]

# MLserver #
    MLServer is an open-source Python library for building production-ready asynchronous APIs for machine learning models.
    https://mlserver.readthedocs.io/en/latest/
    pip install mlserver
    pip install mlserver-sklearn
    
    https://mlserver.readthedocs.io/en/latest/getting-started/index.html


# Model registry

This is used for registering the model in python API.

      result = mlflow.register_model(
          "runs:/d16076a3ec534311817565e6527539c0/sklearn-model", "sk-learn-random-forest-reg"
      )
