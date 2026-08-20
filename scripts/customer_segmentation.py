"""
Segments customers using RFM analysis (Recency, Frequency, Monetary) and
KMeans clustering.

Recency is measured against the day after the last transaction in the
whole dataset, so a recency of 0 means the customer bought something on
the very last day of data we have.

Run scripts/clean_data.py first so data/processed/purchases_clean.csv exists.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")


def build_rfm_table():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "purchases_clean.csv"), parse_dates=["invoice_date"])

    reference_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    print(f"Reference date for recency: {reference_date.date()}")

    rfm = df.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (reference_date - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    return rfm


def choose_cluster_count(X_scaled):
    scores = {}
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"k={k}  silhouette score={score:.3f}")

    # k=2 usually gets the top silhouette score here, but it just splits
    # customers into "low value" and "high value" which is not very useful
    # for a business segmentation, so we pick the best k among 3-6 instead
    # and just note the k=2 score for transparency
    scores_from_3 = {k: v for k, v in scores.items() if k >= 3}
    best_k = max(scores_from_3, key=scores_from_3.get)
    return best_k, scores


def name_segment(avg_recency, avg_monetary, recency_median, monetary_median):
    # a simple 2x2 read on recency vs monetary value, using the overall
    # customer medians as the dividing line for each cluster's average
    if avg_monetary >= monetary_median and avg_recency <= recency_median:
        return "High Value"
    if avg_monetary >= monetary_median and avg_recency > recency_median:
        return "At Risk"
    if avg_monetary < monetary_median and avg_recency <= recency_median:
        return "Occasional"
    return "Lapsed"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    rfm = build_rfm_table()
    print(f"\nRFM table built for {len(rfm)} customers")
    print(rfm[["recency", "frequency", "monetary"]].describe())

    # frequency and monetary are heavily right skewed, log transform helps
    # kmeans treat distances more fairly
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    rfm["log_monetary"] = np.log1p(rfm["monetary"].clip(lower=0))
    rfm["log_recency"] = np.log1p(rfm["recency"])

    features = rfm[["log_recency", "log_frequency", "log_monetary"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print("\nChecking cluster counts 2 through 6:")
    best_k, scores = choose_cluster_count(X_scaled)
    print(f"\nk=2 has the top silhouette score but only separates low value from high")
    print(f"value customers, going with k={best_k} instead for more useful segments")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).reset_index()
    print("\nCluster summary:")
    print(cluster_summary)

    recency_median = rfm["recency"].median()
    monetary_median = rfm["monetary"].median()
    cluster_summary["segment_name"] = cluster_summary.apply(
        lambda row: name_segment(row["avg_recency"], row["avg_monetary"], recency_median, monetary_median),
        axis=1,
    )
    print("\nNamed segments:")
    print(cluster_summary[["cluster", "segment_name", "customers", "avg_recency", "avg_frequency", "avg_monetary"]])

    segment_map = dict(zip(cluster_summary["cluster"], cluster_summary["segment_name"]))
    rfm["segment_name"] = rfm["cluster"].map(segment_map)

    customer_table = rfm[["customer_id", "recency", "frequency", "monetary", "cluster", "segment_name"]]
    customer_table.to_csv(os.path.join(TABLES_DIR, "customer_segments.csv"), index=False)
    print("\nSaved outputs/tables/customer_segments.csv")

    cluster_summary.to_csv(os.path.join(TABLES_DIR, "customer_segment_summary.csv"), index=False)
    print("Saved outputs/tables/customer_segment_summary.csv")

    # plot: silhouette score by k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    fig.savefig(os.path.join(FIGURES_DIR, "14_segmentation_silhouette_scores.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/14_segmentation_silhouette_scores.png")

    # plot: customers in frequency vs monetary space, colored by segment
    fig, ax = plt.subplots(figsize=(8, 6))
    for segment in rfm["segment_name"].unique():
        subset = rfm[rfm["segment_name"] == segment]
        ax.scatter(subset["frequency"], subset["monetary"], label=segment, alpha=0.5, s=15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Customer Segments (Frequency vs Monetary)")
    ax.set_xlabel("Frequency (orders, log scale)")
    ax.set_ylabel("Monetary (total spend, log scale)")
    ax.legend()
    fig.savefig(os.path.join(FIGURES_DIR, "15_customer_segments_scatter.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/15_customer_segments_scatter.png")

    # plot: segment sizes
    counts = rfm["segment_name"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Customers per Segment")
    ax.set_ylabel("Number of Customers")
    fig.savefig(os.path.join(FIGURES_DIR, "16_segment_sizes.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("Saved outputs/figures/16_segment_sizes.png")


if __name__ == "__main__":
    main()
