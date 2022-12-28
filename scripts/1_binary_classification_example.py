import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, make_scorer
import sys
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pd.options.mode.chained_assignment = None

session = requests.Session()
session.verify = False


def acquire_data() -> pd.DataFrame:
    """Acquire the data.
    Returns:
        The dataset as a pandas DataFrame.
    """
    return sns.load_dataset("penguins")


def do_feature_engineering(input_data: pd.DataFrame) -> pd.DataFrame:
    """Example of feature engineering.
    Args:
        input_data: Input data.
    Returns:
        A Pandas DataFrame of the feature data.
    """
    # Create a SimpleImputer Class
    imputer = SimpleImputer(missing_values=np.NaN, strategy="median")

    # Fit the columns to the object
    columns = ["bill_depth_mm", "bill_length_mm", "flipper_length_mm", "body_mass_g"]
    imputer = imputer.fit(input_data[columns])

    # Transform the DataFrames column with the fitted data
    input_data[columns] = imputer.transform(input_data[columns])

    # One-hot Encoding the Island Feature
    one_hot = OneHotEncoder()
    encoded = one_hot.fit_transform(input_data[["island"]])
    input_data[one_hot.categories_[0]] = encoded.toarray()

    # One-hot Encoding the Species Feature
    encoded = one_hot.fit_transform(input_data[["species"]])
    input_data[one_hot.categories_[0]] = encoded.toarray()

    input_data["sex"] = input_data["sex"].fillna("Unknown")
    input_data = input_data[input_data["sex"] != "Unknown"]

    input_data["sex_int"] = input_data.loc[:, "sex"].map({"Male": 0, "Female": 1})

    return input_data


def train_and_log_model(input_data: pd.DataFrame) -> None:
    """Run an experiment and log to file using the eval metrics
    Args:
        input_data: Original input data.
    """
    X = input_data.drop(["sex", "sex_int", "species", "island"], axis=1)
    y = input_data["sex_int"]

    logger.info("Building model")
    model = build_model()

    logger.info("Evaluating model metrics with 5 fold cross validation")
    evaluate_model(model, X, y)


def build_model() -> LogisticRegression:
    """Build the model.
    Returns:
        The model object.
    """
    lr = LogisticRegression(max_iter=2500)

    return lr


def evaluate_model(
    clf: Any, input_features: pd.DataFrame, target_variable: pd.DataFrame
) -> None:
    """Evaluate the model and calculate model performance metrics.
    Args:
        clf: the classifier model object
        input_features: input data
        target_variable: target variable - ground truth
    """
    metrics = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average="micro"),
    }
    scores = cross_validate(
        clf,
        input_features,
        target_variable,
        cv=5,
        scoring=metrics,
        return_train_score=True,
    )
    mean_train_accuracy = scores["train_accuracy"].mean()
    mean_train_f1 = scores["train_f1"].mean()
    mean_test_accuracy = scores["test_accuracy"].mean()
    mean_test_f1 = scores["test_f1"].mean()

    logger.info(
        "5-Fold Cross Validated Training Accuracy: {}".format(
            f"{mean_train_accuracy:.3f}"
        )
    )
    logger.info("5-Fold Cross Validated Training F1: {}".format(f"{mean_train_f1:.3f}"))
    logger.info(
        "5-Fold Cross Validated Testing Accuracy: {}".format(
            f"{mean_test_accuracy:.3f}"
        )
    )
    logger.info("5-Fold Cross Validated Testing F1: {}".format(f"{mean_test_f1:.3f}"))


def run_pipeline() -> None:
    """Train and log the model metrics"""
    # Get data
    logger.info("Getting data")
    input_data = acquire_data()
    # Feature engineering
    logger.info("Doing feature engineering")
    features = do_feature_engineering(input_data)
    logger.debug("features shape {}".format(features.shape))
    # Train model and log in mlflow
    train_and_log_model(features)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
