#!/usr/bin/env python
"""
This script trains a Random Forest and generate the inference artifact
and upload it to the Weights and Biases.

"""
import argparse
import logging
import os
import shutil
import json
import matplotlib.pyplot as plt
import tempfile

import mlflow

import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), 
    it returns the delta in days
    between each date and the most recent date in its column

    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(
        lambda d: (
            d.max() - d).dt.days,
        axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    """
    this is the main function which run.
    fetch the training data artifact and divide the data into
    train and validation.
    At the last this will save and upload the inference
    artifact to wandb

    """
    print(sklearn.__version__)

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fixing the random seed for the Random Forest, so we get reproducible
    # results
    rf_config['random_state'] = args.random_seed

    # fetching the training data
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(trainval_local_path)
    # this removes the column "price" from X and puts it into y
    y = X.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # splitting the data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    # getting our pipeline which we used to train our model
    sk_pipe, processed_features = get_inference_pipeline(
        rf_config, args.max_tfidf_features)

    logger.info("Fitting")

    # fitting our model with the train data
    sk_pipe.fit(X_train[processed_features], y_train)

    # Computing r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val[processed_features], y_val)

    y_pred = sk_pipe.predict(X_val[processed_features])
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Saving model package in the MLFlow sklearn format
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Creating the signature variable
    signature = infer_signature(X_val[processed_features], y_pred)

    # Creating the temporary directory "random_forest_dir" which will be deleted
    # at the end of this block
    with tempfile.TemporaryDirectory() as temp_dire:

        # generating path to the "random_forest_dir" directory where
        # we save the model
        export_path = os.path.join(temp_dire, "random_forest_dir")

        # saving our scikit-learn model to the export path
        mlflow.sklearn.save_model(
            sk_pipe,
            export_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.head())

        # Creating a new artifact which store our model
        Artifact = wandb.Artifact(
            args.output_artifact,
            args.artifact_type,
            args.artifact_description,
            metadata=rf_config
        )

        # adding the path of the directory where we stored our model
        Artifact.add_dir(export_path)

        # uploading our artifact to wandb
        run.log_artifact(Artifact)

        Artifact.wait()

    # Plotting feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # Storing the r squared metric to the summary section
    run.summary['r2'] = r_squared

    # Storing the metrics mean absolute error to the summary section
    run.summary["mae"] = mae

    # Upload the visualization to wandb
    run.log(
        {
            "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe, feat_names):

    """
    In this function we are plotting feature importance plot
    and the uploading it to the wandb.

    """
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest"].feature_importances_[
        : len(feat_names) - 1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(
        pipe["random_forest"].feature_importances_[
            len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp,
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):

    """
    In this function we are acreating our inference pipeline and we use column transformer 
    to transform our column and impute missing values this will then make a pipeline
    with the preprocessor and the model.
    this function will return the pipe and the column that is going to be used
    in the model.

    """

    # Selecting ordinal categorical value
    ordinal_categorical = ["room_type"]

    non_ordinal_categorical = ["neighbourhood_group"]

    # Initializing ordinal encoder for ordinal categorical columns
    ordinal_categorical_preproc = OrdinalEncoder()

    # Making a pipeline for non ordinal column neighbourhood_group
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )

    # Let's impute the numerical columns to make sure we can handle missing
    # values
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="mean")

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(
            strategy='constant',
            fill_value='2010-01-01'),
        FunctionTransformer(
            delta_date_feature,
            check_inverse=False,
            validate=False))

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    # Creating the column transformer and applying the respective pipeline to
    # the particular column
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat",
             non_ordinal_categorical_preproc,
             non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + \
        non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    # Initializing the random forest model
    random_Forest = RandomForestRegressor(**rf_config)

    # creating the pipeline with the preprocessor nd the random forest model
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_Forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation",
        required=True)

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
        required=True)

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        help="type of the artifact",
        type=str,
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        help="description of the artifact",
        type=str,
        required=True
    )

    args = parser.parse_args()

    go(args)
