from mlflow_experiment_utils import create_mlflow_experiment

if __name__=="__main__":
    experiment_id = create_mlflow_experiment(experiment_name="testing_mlflow_experiment_01", 
                             artifact_location="testing_mlflow_experiment_artifacts",tags={"env":"dev","version":"1.0.0"})
    print(experiment_id)