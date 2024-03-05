import mlflow

if __name__ == '__main__':
    mlflow.set_experiment("mlflow-experiment")
    mlflow.start_run()
    mlflow.log_param("param1", 1)
    mlflow.log_param("param2", 2)
    mlflow.log_metric("metric1", 1)
    mlflow.log_metric("metric2", 2)
    mlflow.end_run()