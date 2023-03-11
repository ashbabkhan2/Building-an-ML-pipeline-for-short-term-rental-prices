"""
This is our main file which run all of our component in a particular workflow
we only have to run this file and this will create an inference artifact in wandb
when the execution of this file finishes.

Date: 5/March/2023
Author: ashbab khan

"""
import json
import tempfile
import os
import mlflow
import hydra
from omegaconf import DictConfig

# By default steps is set to "all" it means all component is run one
# by one but we have option to run a particular component by passing
# the steps value to hydra such as download, data_split etc.

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest"
]

# This automatically reads in the configuration


@hydra.main(version_base="1.1", config_name='config')
def go(config: DictConfig):

    """
    This is our main function which run all the component all we have to do is 
    to run this file using the command mlflow run . and then it will run all our 
    component it first run get_data component then basic_cleaning and then so on
    at the last our inference pipeline components runs which generate inference artifact.

    """

    # Setting up the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    root_dir = hydra.utils.get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory():

        # ================ get_data component ===============

        if "download" in active_steps:
            # running the get_data comnponent
            _ = mlflow.run(
                os.path.join(root_dir, "components/get_data"),
                "main",
                # version = 'main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": config["parameters"]["download"]["artifact_name"],
                    "artifact_type": config["parameters"]["download"]["artifact_type"],
                    "artifact_description": config["parameters"]["download"]["artifact_description"]
                },
            )

# ============ basic_cleanings component ===========

        if "basic_cleaning" in active_steps:
            # running the basic_cleaning component
            _ = mlflow.run(
                os.path.join(root_dir, "src", "basic_cleaning"),
                "main",
                # version = "main"
                parameters={
                    "input_artifact": config["parameters"]["basic_cleaning"]["input_artifact"],
                    "output_artifact": config["parameters"]["basic_cleaning"]["output_artifact"],
                    "artifact_type": config["parameters"]["basic_cleaning"]["artifact_type"],
                    "artifact_description":
                      config["parameters"]["basic_cleaning"]["artifact_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

# ============== data_check component ================

        if "data_check" in active_steps:
            # running the data_check component
            _ = mlflow.run(
                os.path.join(root_dir, "src", "data_check"),
                "main",
                # version = "main"
                parameters={
                    "csv": config["parameters"]["data_check"]["csv"],
                    "ref": config["parameters"]["data_check"]["ref"],
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                }
            )

# ============== data_split component ================

        if "data_split" in active_steps:
            # running the data_split component
            _ = mlflow.run(
                os.path.join(root_dir, "components", "train_val_test_split"),
                "main",
                # version = "main"
                parameters={
                    "input": config["parameters"]["data_split"]["input"],
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )


# ======== train_random_forest component =======

        if "train_random_forest" in active_steps:

            # we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(
                    dict(
                        config["modeling"]["random_forest"].items()),
                    fp)

            # running the train_random_forest component
            _ = mlflow.run(
                os.path.join(root_dir, "src", "train_random_forest"),
                "main",
                # version = "main"
                parameters={
                    "trainval_artifact": 
                     config["parameters"]["train_random_forest"]["trainval_artifact"],
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": 
                     config["parameters"]["train_random_forest"]["output_artifact"],
                    "artifact_type": config["parameters"]["train_random_forest"]["artifact_type"],
                    "artifact_description":
                      config["parameters"]["train_random_forest"]["artifact_description"]
                }
            )

# ====== test_regression_model component =====

        if "test_regression_model" in active_steps:
            # running the test_regression_model
            _ = mlflow.run(
                os.path.join(root_dir, "components", "test_regression_model"),
                "main",
                # version = "main"
                parameters={
                    "mlflow_model": config["parameters"]["test_regression_model"]["mlflow_model"],
                    "test_dataset": config["parameters"]["test_regression_model"]["test_dataset"]
                }
            )


# ================================================

if __name__ == "__main__":
    go()
