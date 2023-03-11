"""
This script upload our datasets to Weights & Biases

Date: 04/March/2023
Developer: ashbab khan

"""
# Importing packages
import argparse
import logging
import os

import wandb

# Initialzing a logger so that we get the status of our run
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    """
    this function upload the sample1.csv data as an artifact to weights & biases
    and upload the configuration such as the argument passed into this file 

    """

    # Creating a run by initiliazing the wandb.
    # We already set the project and experiment name in the main.py file.  
    run = wandb.init(job_type="download_file")

    # uploading the configurations to the run
    run.config.update(args)

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")

    # Creating artifact
    artifact = wandb.Artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
    )

    # Adding our datasets which is sample1.csv to the artifact
    artifact.add_file(os.path.join("data",args.sample))
    
    # Uploading the artifact to Weghts & Biases
    run.log_artifact(artifact)


if __name__ == "__main__":

    """
    This is the parser area and this catches the argument coming from cmd or MLProject
    then we pass this argument to our go() function as args.

    this python file takes 4 parameters from the MLProject and all are compulsory
    that's why we included a keyword required = true

      1. sample ( required )
      2. artifact name ( required )
      3. artifact type ( required )
      4. artifact description ( required )

    """
    
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("--sample", 
                        type=str,
                        help="Name of the sample to download",
                        required=True)

    parser.add_argument("--artifact_name",
                         type=str,
                         help="Name for the output artifact",
                         required=True)

    parser.add_argument("--artifact_type",
                         type=str,
                         help="Output artifact type.",
                         required=True)

    parser.add_argument("--artifact_description",
                         type=str,
                         help="A brief description of this artifact",
                         required=True
    )

    args = parser.parse_args()

    go(args)
