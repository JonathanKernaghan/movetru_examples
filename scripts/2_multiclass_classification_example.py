import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.4, random_state=0
    )

    logger.info("Building model")
    model = build_model(X_train, y_train)

    logger.info("Evaluating model metrics with training set")
    evaluate_model(model, X_train, y_train, test=False)

    logger.info("Evaluating model metrics with testing set")
    evaluate_model(model, X_test, y_test, test=True)


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    """Build the model.
    Args:
        X_train: DataFrame with features to train.
        y_train: Series with targets for training.
    Returns:
        The trained model object.
    """
    svc = SVC()

    # train model
    svc.fit(X_train, y_train)

    return svc


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
    f1 = f1_score(target_variable, predictions, average="micro")

    logger.info("Accuracy {}".format(accuracy))
    logger.info("F1 Score {}".format(f1))

    if test:
        cm = confusion_matrix(target_variable, predictions)
        f = sns.heatmap(cm, annot=True).set(
            title="Confusion Matrix for Test Data Predictions"
        )
        cm_path = "2_multiclass_classification_basic_example_confusion_matrix.png"
        logger.info("Saving confusion matrix to {}".format(cm_path))
        plt.savefig(cm_path)
        report_path = "2_multiclass_classification_basic_example_report.json"
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
