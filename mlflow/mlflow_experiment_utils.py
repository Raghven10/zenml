import mlflow
from typing import Any

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags:dict[str, Any])-> str:

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
    return experiment_id