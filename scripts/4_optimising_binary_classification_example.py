# * Copyright (C) Jonathan Kernaghan - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# * Proprietary and confidential
# * Written by Jonathan Kernaghan <jkernaghan272@gmail.com>, December 2022

import json
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
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

    logger.info(
        "Evaluating model metrics with 5 fold cross validation (will take a few minutes)"
    )
    evaluate_model(model, X, y)


def build_model() -> RandomizedSearchCV:
    """Build the model.
    Returns:
        The model object.
    """
    rf = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ["sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=25,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    return rf_random


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
    result = clf.fit(input_features, target_variable)
    best_model = result.best_estimator_

    scores = cross_validate(
        best_model,
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

    metrics = {
        "mean_train_accuracy": f"{mean_train_accuracy:.3f}",
        "mean_train_f": f"{mean_train_f1:.3f}",
        "mean_test_accuracy": f"{mean_test_accuracy:.3f}",
        "mean_test_f1": f"{mean_test_f1:.3f}",
    }

    with open(
        "../results/4_optimising_binary_classification_example_results.txt", "w"
    ) as convert_file:
        convert_file.write(json.dumps(metrics))


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
