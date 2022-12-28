import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    make_scorer,
)
import json
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
    return sns.load_dataset("iris")


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
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    imputer = imputer.fit(input_data[columns])

    # Transform the DataFrames column with the fitted data
    input_data[columns] = imputer.transform(input_data[columns])

    # Encoding the Island Feature for modelling as multiclass target
    target_dict = {c: i for i, c in enumerate(input_data.species.unique())}
    input_data["species_encoded"] = input_data["species"].map(target_dict)

    return input_data


def train_and_log_model(input_data: pd.DataFrame) -> None:
    """Run an experiment and log to file using the eval metrics
    Args:
        input_data: Original input data.
    """
    X = input_data.drop(["species", "species_encoded"], axis=1)
    X.info()
    y = input_data["species_encoded"]

    logger.info("Building model")
    model = build_model()

    logger.info("Evaluating model metrics with 5 fold cross validation")
    evaluate_model(model, X, y)


def build_model() -> SVC:
    """Build the model.
    Returns:
        The model object.
    """
    svc = SVC()

    return svc


def save_json(obj, path):
    with open(path, "w") as jf:
        json.dump(obj, jf)


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
