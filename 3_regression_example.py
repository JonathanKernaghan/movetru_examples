import pandas as pd
import numpy as np
import seaborn as sns
import logging
from typing import Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import json
import sys
from sklearn.model_selection import cross_validate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pd.options.mode.chained_assignment = None


def acquire_data() -> pd.DataFrame:
    """Acquire the data.
    Returns:
        The dataset as a pandas DataFrame.
    """
    return sns.load_dataset("tips")


def do_feature_engineering(input_data: pd.DataFrame) -> pd.DataFrame:
    """Example of feature engineering.
    Args:
        input_data: Input data.
    Returns:
        A Pandas DataFrame of the feature data.
    """

    # Min max scale the total bill feature
    mix_max = MinMaxScaler()
    encoded = mix_max.fit_transform(input_data[["total_bill"]])
    input_data["total_bill_normalised"] = encoded

    # One-hot Encoding the day feature
    one_hot = OneHotEncoder()
    encoded = one_hot.fit_transform(input_data[["day"]])
    input_data[one_hot.categories_[0]] = encoded.toarray()

    # Binary encoding features with two labels
    input_data["sex_int"] = input_data.loc[:, "sex"].map({"Male": 0, "Female": 1})
    input_data["time_int"] = input_data.loc[:, "time"].map({"Lunch": 0, "Dinner": 1})
    input_data["smoker_int"] = input_data.loc[:, "smoker"].map({"No": 0, "Yes": 1})

    return input_data


def train_and_log_model(input_data: pd.DataFrame) -> None:
    """Run an experiment and log to file using the eval metrics
    Args:
        input_df: Original input data.
    """
    X = input_data.drop(["sex", "time", "smoker", "day", "tip", "total_bill"], axis=1)
    y = input_data["tip"]
    print(X.head())

    logger.info("Building model")
    model = build_model()

    logger.info("Evaluating model metrics with 5 fold cross validation")
    evaluate_model(model, X, y)


def build_model() -> LinearRegression:
    """Build the model.
    Returns:
        The  model object.
    """
    lr = LinearRegression()
    return lr


def evaluate_model(clf: Any, input_features: pd.DataFrame, target_variable: pd.DataFrame) -> None:
    """Evaluate the model and calculate model performance metrics.
    Args:
        clf: the classifier model object
        input_features: input data
        target_variable: target variable - ground truth
        test: if we are evaluating the model on the test set or not
    """
    metrics = {"MAE": make_scorer(mean_absolute_error), "MSE": make_scorer(mean_squared_error)}
    scores = cross_validate(clf, input_features, target_variable, cv=5, scoring=metrics)
    mean_mae = np.mean(scores["test_MAE"])
    mean_mse = np.mean(scores["test_MSE"])

    logger.info("5-Fold Cross Validated MAE: {}".format(f"{mean_mae:.2f}"))
    logger.info("5-Fold Cross Validated MSE: {}".format(f"{mean_mse:.2f}"))


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
