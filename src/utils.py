import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path="configs/config.yaml"):
    """Load yaml configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config):
    """
    Load dataset from path specified in config, clean types and extract features/target.
    """
    csv_path = config["data"]["csv_path"]
    target_col = config["data"]["target"]
    num_cols = config["features"]["numeric"]
    cat_cols = config["features"]["categorical"]

    df = pd.read_csv(csv_path)

    # Clean TotalCharges for Telco dataset (convert empty strings/spaces to NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Clean target column (e.g. Yes/No -> 1/0)
    unique_vals = set(df[target_col].dropna().unique())
    if unique_vals in ({"Yes", "No"}, {"yes", "no"}):
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
        df[target_col] = df[target_col].map(mapping).astype(int)

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Perform stratified train-test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
