"""
Stage 1: Data Engineering
Load raw data, clean it (missing values + outliers) and split into train/test.

Run from the project root:
    python code/datasets/data_engineering.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- paths (relative to the project root, so it works from any folder) ---
ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# In these columns 0 is impossible (e.g. blood pressure can't be 0), so 0 = missing value
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET = "Outcome"


def load_data():
    """Step 1. Load the csv file from data/raw"""
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded rows: {len(df)}, columns: {df.shape[1]}")
    return df


def mark_missing(df):
    """Step 2. Replace impossible zeros with NaN (real 'missing' marker)"""
    df = df.copy()
    df[ZERO_IS_MISSING] = df[ZERO_IS_MISSING].replace(0, float("nan"))
    print("Missing values per column:")
    print(df[ZERO_IS_MISSING].isna().sum())
    return df


def remove_outliers(df):
    """Step 3. Remove outliers with the IQR rule.
    A row is dropped if any feature is outside [Q1 - 3*IQR, Q3 + 3*IQR].
    We use 3 instead of 1.5 because the dataset is small and we don't want to lose many rows.
    This is done BEFORE filling missing values: if we filled them with the median first,
    many values would become identical and the IQR would shrink, so too many rows
    would be marked as outliers.
    NaN values are skipped when computing quantiles and are not treated as outliers."""
    features = [c for c in df.columns if c != TARGET]
    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 3 * iqr
    high = q3 + 3 * iqr

    inside = (df[features] >= low) & (df[features] <= high)
    keep = (inside | df[features].isna()).all(axis=1)
    print(f"Removed outliers: {(~keep).sum()} rows")
    return df[keep]


def fill_missing(df):
    """Step 4. Fill missing values with the median (median is robust to outliers)"""
    df = df.copy()
    for col in ZERO_IS_MISSING:
        df[col] = df[col].fillna(df[col].median())
    return df


def split_and_save(df):
    """Step 5. Split 80/20 and save to data/processed"""
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],  # keep the same share of diabetic patients in train and test
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")
    print(f"Saved to {PROCESSED_DIR}")


def main():
    df = load_data()
    df = mark_missing(df)
    df = remove_outliers(df)
    df = fill_missing(df)
    split_and_save(df)


if __name__ == "__main__":
    main()