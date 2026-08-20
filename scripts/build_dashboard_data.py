"""
Builds small aggregated JSON files for the static dashboard under docs/data/.

This script reads the csv outputs already produced by the rest of the
pipeline (outputs/tables/, outputs/metrics/) and reshapes them into a
handful of small JSON files. It does not read raw transaction data, the
sqlite database, or any customer-level table, the dashboard is public so
it only gets small aggregated numbers.

Run this after the rest of the pipeline has produced its outputs.
"""

import os
import json
import math
import pandas as pd

TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")
DATA_DIR = os.path.join("docs", "data")

# stock codes that are postage, discounts, manual entries and similar,
# not real products, same list used in scripts/clean_data.py
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "D", "S", "BANK CHARGES", "ADJUST",
    "AMAZONFEE", "DCGSSGIRL", "DCGSSBOY", "PADS", "CRUK", "B",
    "GIFT", "DCGSLBOY", "DCGSLGIRL",
}


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}. Run the pipeline first.")


def require_columns(df, columns, source_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing expected columns: {missing}")


def clean_number(value):
    """turns a pandas/numpy number into a plain python number, and turns
    NaN into None so it serializes as JSON null instead of breaking"""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def clean_records(df):
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        cleaned.append({k: clean_number(v) for k, v in row.items()})
    return cleaned


def load_metrics():
    require_file(METRICS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    require_columns(metrics, ["model", "metric", "value"], "model_metrics.csv")
    if not pd.api.types.is_numeric_dtype(metrics["value"]):
        raise ValueError("model_metrics.csv has non-numeric values in the value column")
    return metrics


def get_metric(metrics, model, metric):
    row = metrics[(metrics["model"] == model) & (metrics["metric"] == metric)]
    if row.empty:
        return None
    return clean_number(row["value"].iloc[0])


def find_model_variant(metrics, prefix, exclude):
    """finds the one model name in model_metrics.csv that starts with the
    given prefix but is not the excluded baseline name, used because the
    winning model name changes depending on the validation result"""
    names = metrics.loc[metrics["model"].str.startswith(prefix), "model"].unique()
    names = [n for n in names if n != exclude]
    return names[0] if names else None


def build_overview(metrics):
    total_revenue = pd.read_csv(os.path.join(TABLES_DIR, "total_revenue.csv"))
    order_count = pd.read_csv(os.path.join(TABLES_DIR, "order_count.csv"))
    active_customers = pd.read_csv(os.path.join(TABLES_DIR, "active_customers.csv"))
    aov = pd.read_csv(os.path.join(TABLES_DIR, "average_order_value.csv"))
    repeat_rate = pd.read_csv(os.path.join(TABLES_DIR, "repeat_customer_rate.csv"))
    cancellation = pd.read_csv(os.path.join(TABLES_DIR, "cancellation_rate.csv"))

    overview = {
        "total_revenue": clean_number(total_revenue["total_revenue"].iloc[0]),
        "orders": clean_number(order_count["order_count"].iloc[0]),
        "active_customers": clean_number(active_customers["active_customers"].iloc[0]),
        "average_order_value": clean_number(aov["average_order_value"].iloc[0]),
        "repeat_customer_rate": clean_number(repeat_rate["repeat_customer_pct"].iloc[0]),
        "cancellation_rate": clean_number(cancellation["cancellation_pct"].iloc[0]),
    }
    for key, value in overview.items():
        if value is None:
            raise ValueError(f"overview metric '{key}' is missing or NaN")
    return overview


def dedupe_products(df, code_col, value_col):
    # a handful of stock codes have more than one description in the raw
    # data, group them back into one row so the same product does not
    # show up twice in a top-products chart
    grouped = df.groupby(code_col).agg(
        description=("description", "first"),
        value=(value_col, "sum"),
    ).reset_index()
    return grouped.sort_values("value", ascending=False)


def build_sales(metrics):
    monthly_revenue = pd.read_csv(os.path.join(TABLES_DIR, "monthly_revenue.csv"))
    require_columns(monthly_revenue, ["year_month", "monthly_revenue", "order_count"], "monthly_revenue.csv")
    monthly_revenue["year"] = monthly_revenue["year_month"].str.slice(0, 4)
    years = sorted(monthly_revenue["year"].unique().tolist())

    revenue_by_country = pd.read_csv(os.path.join(TABLES_DIR, "revenue_by_country.csv"))
    require_columns(revenue_by_country, ["country", "total_revenue", "order_count"], "revenue_by_country.csv")
    revenue_by_country = revenue_by_country.sort_values("total_revenue", ascending=False).head(15)

    top_revenue = pd.read_csv(os.path.join(TABLES_DIR, "top_products_by_revenue.csv"))
    require_columns(top_revenue, ["stock_code", "description", "total_revenue"], "top_products_by_revenue.csv")
    top_revenue = top_revenue[~top_revenue["stock_code"].isin(NON_PRODUCT_CODES)]
    top_revenue = dedupe_products(top_revenue, "stock_code", "total_revenue").head(10)

    top_quantity = pd.read_csv(os.path.join(TABLES_DIR, "top_products_by_quantity.csv"))
    require_columns(top_quantity, ["stock_code", "description", "total_units_sold"], "top_products_by_quantity.csv")
    top_quantity = top_quantity[~top_quantity["stock_code"].isin(NON_PRODUCT_CODES)]
    top_quantity = dedupe_products(top_quantity, "stock_code", "total_units_sold").head(10)

    return {
        "monthly_revenue": clean_records(monthly_revenue[["year_month", "monthly_revenue", "order_count"]]),
        "years": years,
        "revenue_by_country": clean_records(revenue_by_country),
        "top_products_by_revenue": clean_records(
            top_revenue.rename(columns={"value": "total_revenue"})[["stock_code", "description", "total_revenue"]]
        ),
        "top_products_by_quantity": clean_records(
            top_quantity.rename(columns={"value": "total_units_sold"})[["stock_code", "description", "total_units_sold"]]
        ),
    }


def build_churn_summary(metrics):
    return {
        "logistic_regression": {
            "precision": get_metric(metrics, "churn_logistic_regression", "precision"),
            "recall": get_metric(metrics, "churn_logistic_regression", "recall"),
            "f1": get_metric(metrics, "churn_logistic_regression", "F1"),
            "roc_auc": get_metric(metrics, "churn_logistic_regression", "ROC_AUC"),
        },
        "random_forest": {
            "precision": get_metric(metrics, "churn_random_forest", "precision"),
            "recall": get_metric(metrics, "churn_random_forest", "recall"),
            "f1": get_metric(metrics, "churn_random_forest", "F1"),
            "roc_auc": get_metric(metrics, "churn_random_forest", "ROC_AUC"),
        },
    }


def build_customers(metrics):
    segment_summary = pd.read_csv(os.path.join(TABLES_DIR, "customer_segment_summary.csv"))
    require_columns(
        segment_summary,
        ["cluster", "segment_name", "customers", "avg_recency", "avg_frequency", "avg_monetary"],
        "customer_segment_summary.csv",
    )

    churn = build_churn_summary(metrics)

    feature_importance_path = os.path.join(TABLES_DIR, "churn_feature_importance.csv")
    feature_importance = []
    if os.path.exists(feature_importance_path):
        fi = pd.read_csv(feature_importance_path)
        require_columns(fi, ["feature", "coefficient"], "churn_feature_importance.csv")
        feature_importance = clean_records(fi)

    return {
        "segments": clean_records(segment_summary),
        "churn": churn,
        "churn_feature_importance": feature_importance,
    }


def build_models(metrics):
    sales_best = find_model_variant(metrics, "sales_forecast_", "sales_forecast_baseline")
    demand_best = find_model_variant(metrics, "demand_", "demand_baseline")

    sales_best_name = sales_best.replace("sales_forecast_", "") if sales_best else None
    demand_best_name = demand_best.replace("demand_", "") if demand_best else None

    sales_results = pd.read_csv(os.path.join(TABLES_DIR, "sales_forecast_results.csv"))
    require_columns(
        sales_results,
        ["week_start", "actual_revenue", "predicted_revenue", "naive_prediction"],
        "sales_forecast_results.csv",
    )

    demand_results = pd.read_csv(os.path.join(TABLES_DIR, "demand_prediction_results.csv"))
    require_columns(
        demand_results,
        ["stock_code", "description", "week_start", "actual_quantity", "predicted_quantity", "naive_prediction"],
        "demand_prediction_results.csv",
    )
    demand_products = {}
    for stock_code, group in demand_results.groupby("stock_code"):
        group = group.sort_values("week_start")
        demand_products[stock_code] = {
            "description": group["description"].iloc[0],
            "weeks": group["week_start"].tolist(),
            "actual": [clean_number(v) for v in group["actual_quantity"]],
            "predicted": [clean_number(v) for v in group["predicted_quantity"]],
            "naive": [clean_number(v) for v in group["naive_prediction"]],
        }

    return {
        "sales_forecast": {
            "model_name": sales_best_name,
            "baseline_mae": get_metric(metrics, "sales_forecast_baseline", "MAE"),
            "baseline_rmse": get_metric(metrics, "sales_forecast_baseline", "RMSE"),
            "model_mae": get_metric(metrics, sales_best, "MAE") if sales_best else None,
            "model_rmse": get_metric(metrics, sales_best, "RMSE") if sales_best else None,
            "test_weeks": clean_records(sales_results),
        },
        "demand_prediction": {
            "model_name": demand_best_name,
            "baseline_mae": get_metric(metrics, "demand_baseline", "MAE"),
            "baseline_rmse": get_metric(metrics, "demand_baseline", "RMSE"),
            "model_mae": get_metric(metrics, demand_best, "MAE") if demand_best else None,
            "model_rmse": get_metric(metrics, demand_best, "RMSE") if demand_best else None,
            "products": demand_products,
        },
        "price_estimation": {
            "baseline_mae": get_metric(metrics, "price_baseline", "MAE"),
            "baseline_rmse": get_metric(metrics, "price_baseline", "RMSE"),
            "model_mae": get_metric(metrics, "price_prediction", "MAE"),
            "model_rmse": get_metric(metrics, "price_prediction", "RMSE"),
            "model_r2": get_metric(metrics, "price_prediction", "R2"),
        },
        "churn": build_churn_summary(metrics),
    }


def build_recommendations(metrics):
    similar = pd.read_csv(os.path.join(TABLES_DIR, "recommendation_similar_products_examples.csv"))
    require_columns(
        similar,
        ["source_stock_code", "source_description", "stock_code", "description", "similarity"],
        "recommendation_similar_products_examples.csv",
    )

    products = {}
    for source_code, group in similar.groupby("source_stock_code"):
        recs = []
        for _, row in group.iterrows():
            recs.append({
                "stock_code": row["stock_code"],
                "description": row["description"],
                "similarity": clean_number(round(row["similarity"], 3)),
            })
        products[source_code] = {
            "description": group["source_description"].iloc[0],
            "recommendations": recs,
        }

    return {
        "products": products,
        "popularity_baseline_hit_rate": get_metric(metrics, "recommendation_popularity_baseline", "HitRate@5"),
        "item_item_hit_rate": get_metric(metrics, "recommendation_item_item", "HitRate@5"),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    metrics = load_metrics()

    outputs = {
        "overview.json": build_overview(metrics),
        "sales.json": build_sales(metrics),
        "customers.json": build_customers(metrics),
        "models.json": build_models(metrics),
        "recommendations.json": build_recommendations(metrics),
    }

    for filename, payload in outputs.items():
        path = os.path.join(DATA_DIR, filename)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"Saved {path}")

    print("\nDashboard data build complete")


if __name__ == "__main__":
    main()
