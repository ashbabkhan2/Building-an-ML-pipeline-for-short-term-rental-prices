"""
This step takes the best model, tagged with the "prod" tag, and 
tests it against the test dataset

Date: 09/March/2023
Developer: ashbab khan

"""

# Importing the packages 
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    """
    This function fetch two artifacts 
      1: Inference artifact ( our saved ML model )
      2: Test data artifact ( test datasets )

    then we load the ML model and score it against our test datasets
    the we store the metrics such as r2 and mae to the run summary of wandb

    """

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    
    # Downloading the inference artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    logger.info("fetching the test artifact path")
    # Downloading test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    # Loading the ML model
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    # Scoring the model
    r_squared = sk_pipe.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    # Log MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


if __name__ == "__main__":

    """
    This is the parser area and this catches the argument coming from cmd or MLProject
    then we pass this argument to our go() function as args.

    this python file takes 2 parameters from the MLProject and all are compulsory
    that's why we included a keyword required = true

      1. mlflow_model ( required )
      2. test_dataset ( required )
      
    """

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
