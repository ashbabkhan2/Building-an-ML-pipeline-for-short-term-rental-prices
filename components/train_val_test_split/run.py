"""
This script splits the provided dataframe in test and remainder

Date: 07/March/2023
Developer: ashbab khan

"""

import argparse
import logging
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    this function will fetch the clean data artifact and then split it
    into two artifact train and test based on the test size we pass
    and the saved it into two different artifacts.

    """

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Downloading input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # This loop will generate two artifacts
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            # Saving the csv file
            df.to_csv(fp.name, index=False)

            # Creating a new Artifact
            artifact = wandb.Artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of data"
            )

            # Adding the recently saved csv to the artifact
            artifact.add_file(fp.name)

            # Uploading the artifact to Weights and Biases
            run.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("--input", type=str, help="Input artifact to split")

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False)

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False)

    args = parser.parse_args()

    go(args)
