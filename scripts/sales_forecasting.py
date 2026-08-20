"""
Forecasts total weekly retail revenue.

Builds a weekly revenue series from the cleaned transactions, compares a
naive "last week repeats" baseline against a few regression models
trained on lag and rolling average features.

Model selection uses a validation period, not the final test period. The
final test period is only touched once, after the winning model has
already been chosen. This avoids picking whichever model happens to look
best on the exact weeks we report results for.

Split, in chronological order:
    training period -> validation period -> final test period

Run scripts/clean_data.py first so data/processed/retail_clean.csv exists.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")

TEST_WEEKS = 15
VALIDATION_WEEKS = 12


def build_weekly_revenue():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "retail_clean.csv"), parse_dates=["invoice_date"])
    completed = df[df["is_cancelled"] == False].copy()
    completed["week_start"] = completed["invoice_date"].dt.to_period("W-SUN").dt.start_time

    weekly = completed.groupby("week_start")["revenue"].sum().sort_index()
    weekly = weekly.reset_index()
    weekly.columns = ["week_start", "revenue"]

    # the last week in the dataset is not a full 7 days (data stops on
    # 2011-12-09), so it would understate revenue, drop it
    weekly = weekly.iloc[:-1].reset_index(drop=True)
    return weekly


def add_features(weekly):
    weekly = weekly.copy()
    weekly["lag_1"] = weekly["revenue"].shift(1)
    weekly["lag_2"] = weekly["revenue"].shift(2)
    weekly["lag_4"] = weekly["revenue"].shift(4)
    weekly["rolling_4"] = weekly["revenue"].shift(1).rolling(4).mean()
    weekly["rolling_8"] = weekly["revenue"].shift(1).rolling(8).mean()
    weekly["month"] = weekly["week_start"].dt.month
    weekly["week_number"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["naive_prediction"] = weekly["lag_1"]

    weekly = weekly.dropna().reset_index(drop=True)
    return weekly


def append_metrics(rows):
    # the winning model name changes depending on the validation result, so
    # clear out any old "sales_forecast_<model>" rows first, otherwise a
    # stale row from a previously winning model would stick around
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    new_rows = pd.DataFrame(rows, columns=["model", "metric", "value"])
    if os.path.exists(METRICS_PATH):
        existing = pd.read_csv(METRICS_PATH)
        existing = existing[~existing["model"].str.startswith("sales_forecast")]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(METRICS_PATH, index=False)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    weekly = build_weekly_revenue()
    print(f"Weekly revenue series has {len(weekly)} complete weeks")

    weekly = add_features(weekly)
    print(f"{len(weekly)} weeks left after building lag features")

    feature_cols = ["lag_1", "lag_2", "lag_4", "rolling_4", "rolling_8", "month", "week_number"]
    X = weekly[feature_cols]
    y = weekly["revenue"]

    test_start = len(weekly) - TEST_WEEKS
    val_start = test_start - VALIDATION_WEEKS

    X_train, X_val, X_test = X.iloc[:val_start], X.iloc[val_start:test_start], X.iloc[test_start:]
    y_train, y_val, y_test = y.iloc[:val_start], y.iloc[val_start:test_start], y.iloc[test_start:]
    naive_test = weekly["naive_prediction"].iloc[test_start:]
    weeks_test = weekly["week_start"].iloc[test_start:]

    print(f"Train weeks: {len(X_train)}, Validation weeks: {len(X_val)}, Test weeks: {len(X_test)}")

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    print("\nValidation results (used to pick the model, not reported as final):")
    val_scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_scores[name] = val_mae
        print(f"{name:<20} validation MAE: {val_mae:,.0f}")

    best_name = min(val_scores, key=val_scores.get)
    print(f"\nBest model on validation: {best_name}")

    # refit the winning model on train + validation, then touch the final
    # test period exactly once
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    best_model = candidates[best_name]
    best_model.fit(X_train_val, y_train_val)
    test_preds = best_model.predict(X_test)

    baseline_mae = mean_absolute_error(y_test, naive_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, naive_test))
    model_mae = mean_absolute_error(y_test, test_preds)
    model_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"\nFinal test results ({TEST_WEEKS} untouched weeks):")
    print(f"Naive baseline       MAE: {baseline_mae:,.0f}  RMSE: {baseline_rmse:,.0f}")
    print(f"{best_name:<20} MAE: {model_mae:,.0f}  RMSE: {model_rmse:,.0f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weeks_test, y_test.values, marker="o", label="Actual")
    ax.plot(weeks_test, test_preds, marker="o", label=f"Predicted ({best_name})")
    ax.plot(weeks_test, naive_test.values, linestyle="--", label="Naive baseline")
    ax.set_title("Weekly Revenue Forecast: Actual vs Predicted (Final Test Weeks)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue (GBP)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "12_sales_forecast_actual_vs_predicted.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/12_sales_forecast_actual_vs_predicted.png")

    results_table = pd.DataFrame({
        "week_start": weeks_test.values,
        "actual_revenue": y_test.values,
        "predicted_revenue": test_preds,
        "naive_prediction": naive_test.values,
    })
    results_table.to_csv(os.path.join(TABLES_DIR, "sales_forecast_results.csv"), index=False)
    print("Saved outputs/tables/sales_forecast_results.csv")

    metric_rows = [
        ["sales_forecast_baseline", "MAE", round(baseline_mae, 2)],
        ["sales_forecast_baseline", "RMSE", round(baseline_rmse, 2)],
        [f"sales_forecast_{best_name}", "MAE", round(model_mae, 2)],
        [f"sales_forecast_{best_name}", "RMSE", round(model_rmse, 2)],
    ]
    append_metrics(metric_rows)
    print("Saved metrics to outputs/metrics/model_metrics.csv")


if __name__ == "__main__":
    main()
