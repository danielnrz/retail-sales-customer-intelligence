"""
Predicts which customers are likely to go inactive.

Important: the Online Retail II dataset has no real churn label. What we
build here is a proxy: a customer counts as "churned" if they made at
least one purchase before a cutoff date but no purchase in the 90 days
after that cutoff. All features are built only from transactions before
the cutoff, so the model never sees the future.

Run scripts/clean_data.py first so the processed csv files exist.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
FUTURE_WINDOW_DAYS = 90

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")


def load_data():
    purchases = pd.read_csv(os.path.join(PROCESSED_DIR, "purchases_clean.csv"), parse_dates=["invoice_date"])
    full = pd.read_csv(os.path.join(PROCESSED_DIR, "retail_clean.csv"), parse_dates=["invoice_date"])
    return purchases, full


def build_features(purchases, full, cutoff_date):
    before = purchases[purchases["invoice_date"] < cutoff_date].copy()
    cancellations_before = full[(full["invoice_date"] < cutoff_date) & (full["is_cancelled"])].copy()

    customers = before["customer_id"].unique()

    features = before.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (cutoff_date - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
        total_quantity=("quantity", "sum"),
        unique_products=("stock_code", "nunique"),
        active_months=("year_month", "nunique"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    ).reset_index()

    features["avg_order_value"] = features["monetary"] / features["frequency"]

    customer_lifetime_days = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["avg_days_between_purchases"] = np.where(
        features["frequency"] > 1,
        customer_lifetime_days / (features["frequency"] - 1),
        features["recency"],
    )

    cancel_counts = cancellations_before.groupby("customer_id")["invoice"].nunique().rename("cancelled_orders")
    features = features.merge(cancel_counts, on="customer_id", how="left")
    features["cancelled_orders"] = features["cancelled_orders"].fillna(0)
    features["cancellation_rate"] = features["cancelled_orders"] / (features["frequency"] + features["cancelled_orders"])

    # most customers are from the UK, this flag separates them from
    # everyone else instead of one-hot encoding 40+ countries
    main_country = before.groupby("customer_id")["country"].agg(lambda x: x.value_counts().index[0])
    features["is_uk"] = features["customer_id"].map(main_country).eq("United Kingdom").astype(int)

    features = features.drop(columns=["first_purchase", "last_purchase"])
    return features


def build_labels(purchases, cutoff_date, future_end_date, customers):
    future = purchases[(purchases["invoice_date"] >= cutoff_date) & (purchases["invoice_date"] < future_end_date)]
    active_in_future = set(future["customer_id"].unique())
    labels = pd.DataFrame({"customer_id": customers})
    labels["churned"] = (~labels["customer_id"].isin(active_in_future)).astype(int)
    return labels


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

    purchases, full = load_data()

    max_date = purchases["invoice_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=FUTURE_WINDOW_DAYS)
    print(f"Data goes up to {max_date.date()}")
    print(f"Cutoff date: {cutoff_date.date()} (features use data before this)")
    print(f"Future window: {cutoff_date.date()} to {max_date.date()} ({FUTURE_WINDOW_DAYS} days)")

    features = build_features(purchases, full, cutoff_date)
    labels = build_labels(purchases, cutoff_date, max_date, features["customer_id"])

    data = features.merge(labels, on="customer_id")
    print(f"\nCustomers with purchase history before cutoff: {len(data)}")
    print(f"Churn rate: {data['churned'].mean():.2%}")

    feature_cols = [
        "recency", "frequency", "monetary", "total_quantity", "unique_products",
        "active_months", "avg_order_value", "avg_days_between_purchases",
        "cancelled_orders", "cancellation_rate", "is_uk",
    ]
    X = data[feature_cols]
    y = data["churned"]

    # random split is fine here, the leakage risk in this task is temporal
    # (future purchases leaking into features) which is already handled by
    # the cutoff date, not by how train/test rows are split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train customers: {len(X_train)}, Test customers: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    log_reg_preds = log_reg.predict(X_test_scaled)
    log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    models = {
        "churn_logistic_regression": (y_test, log_reg_preds, log_reg_probs),
        "churn_random_forest": (y_test, rf_preds, rf_probs),
    }

    metric_rows = []
    for name, (true, pred, prob) in models.items():
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        auc = roc_auc_score(true, prob)
        print(f"\n{name}")
        print(f"  precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}  roc_auc: {auc:.3f}")
        metric_rows.extend([
            [name, "precision", round(precision, 3)],
            [name, "recall", round(recall, 3)],
            [name, "F1", round(f1, 3)],
            [name, "ROC_AUC", round(auc, 3)],
        ])

    append_metrics(metric_rows)
    print("\nSaved metrics to outputs/metrics/model_metrics.csv")

    # confusion matrix for the main model (logistic regression)
    cm = confusion_matrix(y_test, log_reg_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"]).plot(ax=ax, colorbar=False)
    ax.set_title("Churn Prediction Confusion Matrix (Logistic Regression)")
    fig.savefig(os.path.join(FIGURES_DIR, "17_churn_confusion_matrix.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/17_churn_confusion_matrix.png")

    # which features matter most to the logistic regression
    coef_table = pd.DataFrame({"feature": feature_cols, "coefficient": log_reg.coef_[0]})
    coef_table = coef_table.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(coef_table["feature"], coef_table["coefficient"])
    ax.set_title("Logistic Regression Coefficients (Churn Model)")
    ax.set_xlabel("Coefficient (standardized features)")
    fig.savefig(os.path.join(FIGURES_DIR, "18_churn_feature_importance.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/18_churn_feature_importance.png")

    data.to_csv(os.path.join(TABLES_DIR, "churn_features_and_labels.csv"), index=False)
    print("Saved outputs/tables/churn_features_and_labels.csv")


if __name__ == "__main__":
    main()
