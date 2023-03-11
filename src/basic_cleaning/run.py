"""
This module perform basic cleaning such as filtering rows on the basis
of the requirement.

Date: 05/March/2023
Developer: ashbab khan

"""

import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger()


def go(args):
    """
    this function fetch the raw data from wandb and then filter the price column
    on the basis of the fixed range of values ( 10 to 350 )
    and convert the datetime feature from object to datatime using pd.to_datetime().

    """

    logger.info("Initializing a run")
    # Initializing the run
    run = wandb.init(job_type="clean_data")
    run.config.update(args)

    logger.info(f"Using artifact {args.input_artifact}")
    # fetching artifact path
    artifact_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading the file using pd.read_csv() method")
    # reading the data
    data = pd.read_csv(artifact_path)

    logger.info(
        f"Selecting only those data in which price feature is between {args.min_price} \
          and {args.max_price}")
    # filtering the price column to get only rows whose value between 10 to 350
    new_data_boolean = data["price"].between(args.min_price, args.max_price)

    new_data = data[new_data_boolean].copy()

    logger.info(
        f"previously data shape was {data.shape} and after filter it become {new_data.shape}")

    logger.info(
        "Converting our last_review feature from object to datatime datatype")
    # converting last_review column from object to datetime
    new_data["last_review"] = pd.to_datetime(new_data["last_review"])

    # saving the csv file
    new_data.to_csv("clean_sample.csv", index=False)

    logger.info(f"Creating artifact name {args.output_artifact}")
    # Creating artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        args.artifact_type,
        args.artifact_description
    )

    logger.info("Adding clean_sample.csv to the artifact")
    # adding our saved file
    artifact.add_file("clean_sample.csv")

    # uploading the artifact
    run.log_artifact(artifact)
    logger.info("Artifact uploaded")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Cleaning dataset python file"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="our input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="type of the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="description of the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help="minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="maximum price",
        required=True
    )

    args = parser.parse_args()
    go(args)
