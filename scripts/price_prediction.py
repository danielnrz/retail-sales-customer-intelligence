"""
Estimates the unit price of a product from transaction level information.

This is a price ESTIMATION task, not dynamic pricing or price elasticity
modeling. The target is the unit price actually charged, predicted from
the product description, quantity, country, and time of purchase.

The full purchases table has over 700k rows, which is more than needed to
fit a simple Ridge model, so a reproducible sample is used instead. See
SAMPLE_SIZE below.

Run scripts/clean_data.py first so data/processed/purchases_clean.csv exists.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
SAMPLE_SIZE = 60000

PROCESSED_DIR = os.path.join("data", "processed")
TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")


def append_metrics(rows):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    new_rows = pd.DataFrame(rows, columns=["model", "metric", "value"])
    if os.path.exists(METRICS_PATH):
        existing = pd.read_csv(METRICS_PATH)
        existing = existing[~existing["model"].isin(new_rows["model"].unique())]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(METRICS_PATH, index=False)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "purchases_clean.csv"), parse_dates=["invoice_date"])
    print(f"Full purchases table: {len(df):,} rows")

    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
    print(f"Using a reproducible sample of {len(sample):,} rows (random_state={RANDOM_STATE})")

    # keep only the top countries as their own category, group the rest as
    # "Other" so the one-hot encoding does not explode into 40+ columns
    top_countries = sample["country"].value_counts().head(10).index
    sample["country_group"] = sample["country"].where(sample["country"].isin(top_countries), "Other")

    sample["weekday"] = sample["invoice_date"].dt.day_name()

    feature_df = sample[["description", "quantity", "country_group", "month", "weekday", "hour"]]
    target = sample["unit_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, target, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline_prediction = np.full_like(y_test, fill_value=y_train.median(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_prediction)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_prediction))
    print(f"\nMedian price baseline  MAE: {baseline_mae:.3f}  RMSE: {baseline_rmse:.3f}")

    preprocessor = ColumnTransformer(transformers=[
        ("description", TfidfVectorizer(max_features=300, stop_words="english"), "description"),
        ("country", OneHotEncoder(handle_unknown="ignore"), ["country_group"]),
        ("weekday", OneHotEncoder(handle_unknown="ignore"), ["weekday"]),
        ("numeric", StandardScaler(), ["quantity", "month", "hour"]),
    ])

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Ridge price estimation  MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")

    metric_rows = [
        ["price_baseline", "MAE", round(baseline_mae, 3)],
        ["price_baseline", "RMSE", round(baseline_rmse, 3)],
        ["price_prediction", "MAE", round(mae, 3)],
        ["price_prediction", "RMSE", round(rmse, 3)],
        ["price_prediction", "R2", round(r2, 3)],
    ]
    append_metrics(metric_rows)
    print("Saved metrics to outputs/metrics/model_metrics.csv")

    examples = X_test.copy()
    examples["actual_price"] = y_test.values
    examples["predicted_price"] = preds
    examples = examples.sample(n=20, random_state=RANDOM_STATE).sort_index()
    examples.to_csv(os.path.join(TABLES_DIR, "price_prediction_examples.csv"), index=False)
    print("Saved outputs/tables/price_prediction_examples.csv")


if __name__ == "__main__":
    main()
