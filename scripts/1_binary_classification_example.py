import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    logger.info("Building model")
    model = build_model(X_train, y_train)

    logger.info("Evaluating model metrics with training set")
    evaluate_model(model, X_train, y_train, test=False)

    logger.info("Evaluating model metrics with testing set")
    evaluate_model(model, X_test, y_test, test=True)


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Build the model.
    Args:
        X_train: DataFrame with features to train.
        y_train: Series with targets for training.
    Returns:
        The trained model object.
    """
    lr = LogisticRegression()

    # train model
    lr.fit(X_train, y_train)

    return lr


def save_json(obj, path):
    with open(path, "w") as jf:
        json.dump(obj, jf)


def evaluate_model(
    clf: Any, input_features: pd.DataFrame, target_variable: pd.DataFrame, test: bool
) -> None:
    """Evaluate the model and calculate model performance metrics.
    Args:
        clf: the classifier model object
        input_features: input data
        target_variable: target variable - ground truth
        test: if we are evaluating the model on the test set or not
    """
    predictions = clf.predict(input_features)
    accuracy = accuracy_score(target_variable, predictions)
    f1 = f1_score(target_variable, predictions)

    logger.info("Accuracy {}".format(accuracy))
    print("Accuracy =", accuracy)
    logger.info("F1 Score {}".format(f1))
    print("F1 Score =", f1)

    if test:
        cm = confusion_matrix(target_variable, predictions)
        f = sns.heatmap(cm, annot=True).set(
            title="Confusion Matrix for Test Data Predictions"
        )
        cm_path = "1_binary_classification_example_confusion_matrix.png"
        logger.info("Saving confusion matrix to {}".format(cm_path))
        plt.savefig(cm_path)
        report_path = "1_binary_classification_example_report.json"
        logger.info("Saving classification report matrix to {}".format(report_path))
        report = classification_report(target_variable, predictions, output_dict=True)
        save_json(report, report_path)


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
