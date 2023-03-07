import argparse
import wandb
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger()
def go(args):

    logger.info("Initializing a run")
    run = wandb.init(job_type="clean_data")
    run.config.update(args)

    logger.info(f"Using artifact {args.input_artifact}")
    artifact_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Reading the file using pd.read_csv() method")
    data = pd.read_csv(artifact_path)

    logger.info(f"Selecting only those data in which price feature is between {args.min_price} and {args.max_price}")
    new_data_boolean = data["price"].between(args.min_price,args.max_price)
    new_data = data[new_data_boolean].copy()

    logger.info(f"previously data shape was {data.shape} and after filter it become {new_data.shape}")
    
    logger.info(f"Converting our last_review feature from object to datatime datatype")
    new_data["last_review"] = pd.to_datetime(new_data["last_review"])

    new_data.to_csv("clean_sample.csv",index=False)

    logger.info(f"Creating artifact name {args.output_artifact}")
    artifact = wandb.Artifact(
        args.output_artifact,
        args.artifact_type,
        args.artifact_description
    )

    logger.info(f"Adding clean_sample.csv to the artifact")
    artifact.add_file("clean_sample.csv")

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
