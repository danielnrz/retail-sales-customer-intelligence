"""
Predicts next week's unit demand for individual products.

This is different from the weekly revenue forecast, here the goal is to
estimate how many units of a specific product will sell next week. The
full catalog has thousands of products and most of them sell too rarely
to model well, so this script focuses on the top selling products.

Run scripts/clean_data.py first so data/processed/purchases_clean.csv exists.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
N_TOP_PRODUCTS = 50
TEST_WEEKS = 10

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")


def build_product_week_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "purchases_clean.csv"), parse_dates=["invoice_date"])
    df["week_start"] = df["invoice_date"].dt.to_period("W-SUN").dt.start_time

    # drop the last, incomplete week, same reasoning as sales_forecasting.py
    last_week = df["week_start"].max()
    df = df[df["week_start"] < last_week]

    top_products = (
        df.groupby("stock_code")["quantity"].sum().sort_values(ascending=False).head(N_TOP_PRODUCTS).index
    )
    df = df[df["stock_code"].isin(top_products)]

    weekly = (
        df.groupby(["stock_code", "week_start"])
        .agg(quantity=("quantity", "sum"), revenue=("revenue", "sum"), avg_price=("unit_price", "mean"))
        .reset_index()
    )

    # fill in weeks where a product had zero sales, otherwise lag features
    # break across gaps
    all_weeks = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
    filled_parts = []
    for code, group in weekly.groupby("stock_code"):
        numeric_cols = group[["week_start", "quantity", "revenue", "avg_price"]]
        numeric_cols = numeric_cols.set_index("week_start").reindex(all_weeks, fill_value=0)
        numeric_cols["stock_code"] = code
        numeric_cols.index.name = "week_start"
        filled_parts.append(numeric_cols.reset_index())
    weekly_filled = pd.concat(filled_parts, ignore_index=True)

    # description is useful for the summary table, attach the most common one
    descriptions = df.groupby("stock_code")["description"].agg(lambda x: x.value_counts().index[0])
    weekly_filled["description"] = weekly_filled["stock_code"].map(descriptions)

    return weekly_filled


def add_features(weekly):
    weekly = weekly.sort_values(["stock_code", "week_start"]).reset_index(drop=True)
    grouped = weekly.groupby("stock_code")["quantity"]

    weekly["qty_lag_1"] = grouped.shift(1)
    weekly["qty_lag_2"] = grouped.shift(2)
    weekly["qty_lag_4"] = grouped.shift(4)
    weekly["qty_rolling_4"] = grouped.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    weekly["recent_revenue"] = weekly.groupby("stock_code")["revenue"].shift(1)
    weekly["recent_avg_price"] = weekly.groupby("stock_code")["avg_price"].shift(1)
    weekly["month"] = weekly["week_start"].dt.month
    weekly["week_number"] = weekly["week_start"].dt.isocalendar().week.astype(int)

    weekly = weekly.dropna().reset_index(drop=True)
    return weekly


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
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print(f"Building product-week data for the top {N_TOP_PRODUCTS} products...")
    weekly = build_product_week_data()
    weekly = add_features(weekly)
    print(f"{len(weekly)} product-week rows after feature engineering")

    feature_cols = [
        "qty_lag_1", "qty_lag_2", "qty_lag_4", "qty_rolling_4",
        "recent_revenue", "recent_avg_price", "month", "week_number",
    ]

    cutoff_week = weekly["week_start"].sort_values().unique()[-TEST_WEEKS]
    train = weekly[weekly["week_start"] < cutoff_week]
    test = weekly[weekly["week_start"] >= cutoff_week]
    print(f"Train rows: {len(train)}, Test rows: {len(test)} (test starts {cutoff_week.date()})")

    X_train, y_train = train[feature_cols], train["quantity"]
    X_test, y_test = test[feature_cols], test["quantity"]

    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "preds": preds}
        print(f"{name:<20} MAE: {mae:.2f}  RMSE: {rmse:.2f}")

    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]
    print(f"\nBest model: {best_name}")

    test = test.copy()
    test["predicted_quantity"] = best["preds"]

    # show a few representative products: the 3 highest volume ones
    top_3_codes = (
        weekly.groupby("stock_code")["quantity"].sum().sort_values(ascending=False).head(3).index
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    for ax, code in zip(axes, top_3_codes):
        product_test = test[test["stock_code"] == code].sort_values("week_start")
        desc = product_test["description"].iloc[0] if len(product_test) else code
        ax.plot(product_test["week_start"], product_test["quantity"], marker="o", label="Actual")
        ax.plot(product_test["week_start"], product_test["predicted_quantity"], marker="o", label="Predicted")
        ax.set_title(f"{code} - {desc}"[:60])
        ax.set_ylabel("Units")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "13_demand_prediction_examples.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/13_demand_prediction_examples.png")

    example_table = test[["stock_code", "description", "week_start", "quantity", "predicted_quantity"]]
    example_table = example_table[example_table["stock_code"].isin(top_3_codes)].sort_values(["stock_code", "week_start"])
    example_table.to_csv(os.path.join(TABLES_DIR, "demand_prediction_examples.csv"), index=False)
    print("Saved outputs/tables/demand_prediction_examples.csv")

    metric_rows = [
        ["demand_prediction", "MAE", round(best["mae"], 2)],
        ["demand_prediction", "RMSE", round(best["rmse"], 2)],
    ]
    append_metrics(metric_rows)
    print("Saved metrics to outputs/metrics/model_metrics.csv")


if __name__ == "__main__":
    main()
