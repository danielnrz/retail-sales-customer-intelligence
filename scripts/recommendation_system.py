"""
Builds a simple item-item collaborative filtering recommender from customer
purchase history.

The idea: two products are "similar" if the same customers tend to buy
both of them. Once we know which products are similar to each other, we
can recommend products similar to what a customer has already bought.

To keep the customer-product matrix a reasonable size, this script only
considers:
  - customers who have bought at least MIN_PRODUCTS_PER_CUSTOMER
    different products
  - products bought by at least MIN_CUSTOMERS_PER_PRODUCT different
    customers

Run scripts/clean_data.py first so data/processed/purchases_clean.csv exists.
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
MIN_PRODUCTS_PER_CUSTOMER = 5
MIN_CUSTOMERS_PER_PRODUCT = 20
TOP_N = 5
EVAL_HOLDOUT_DAYS = 60

PROCESSED_DIR = os.path.join("data", "processed")
TABLES_DIR = os.path.join("outputs", "tables")
METRICS_PATH = os.path.join("outputs", "metrics", "model_metrics.csv")


def filter_interactions(df):
    products_per_customer = df.groupby("customer_id")["stock_code"].transform("nunique")
    df = df[products_per_customer >= MIN_PRODUCTS_PER_CUSTOMER]

    customers_per_product = df.groupby("stock_code")["customer_id"].transform("nunique")
    df = df[customers_per_product >= MIN_CUSTOMERS_PER_PRODUCT]

    return df


def build_similarity_matrix(df):
    """returns (item_similarity_df, customer_products dict) built from the
    given purchase interactions, using binary purchased/not-purchased"""

    customers = df["customer_id"].astype("category")
    products = df["stock_code"].astype("category")

    rows = products.cat.codes
    cols = customers.cat.codes
    data = np.ones(len(df))

    product_customer_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(products.cat.categories), len(customers.cat.categories)),
    )
    # collapse duplicate (product, customer) pairs down to 1
    product_customer_matrix.data[:] = 1

    similarity = cosine_similarity(product_customer_matrix)
    item_similarity = pd.DataFrame(
        similarity, index=products.cat.categories, columns=products.cat.categories
    )

    customer_products = df.groupby("customer_id")["stock_code"].apply(set).to_dict()

    return item_similarity, customer_products


def recommend_similar_products(item_similarity, descriptions, stock_code, n=TOP_N):
    if stock_code not in item_similarity.index:
        return pd.DataFrame(columns=["stock_code", "description", "similarity"])

    scores = item_similarity.loc[stock_code].drop(index=stock_code)
    top = scores.sort_values(ascending=False).head(n)
    result = pd.DataFrame({"stock_code": top.index, "similarity": top.values})
    result["description"] = result["stock_code"].map(descriptions)
    return result[["stock_code", "description", "similarity"]]


def recommend_for_customer(item_similarity, customer_products, descriptions, customer_id, n=TOP_N):
    bought = customer_products.get(customer_id, set())
    bought = [code for code in bought if code in item_similarity.index]
    if not bought:
        return pd.DataFrame(columns=["stock_code", "description", "score"])

    scores = item_similarity.loc[bought].sum(axis=0)
    scores = scores.drop(index=bought, errors="ignore")
    top = scores.sort_values(ascending=False).head(n)
    result = pd.DataFrame({"stock_code": top.index, "score": top.values})
    result["description"] = result["stock_code"].map(descriptions)
    return result[["stock_code", "description", "score"]]


def evaluate_hit_rate(df):
    cutoff = df["invoice_date"].max() - pd.Timedelta(days=EVAL_HOLDOUT_DAYS)
    train = df[df["invoice_date"] < cutoff]
    test = df[df["invoice_date"] >= cutoff]

    train = filter_interactions(train)
    if train.empty:
        print("Not enough training interactions to evaluate")
        return None

    item_similarity, customer_products = build_similarity_matrix(train)
    descriptions = train.groupby("stock_code")["description"].agg(lambda x: x.value_counts().index[0])

    test_products_by_customer = test.groupby("customer_id")["stock_code"].apply(set).to_dict()

    eval_customers = [c for c in customer_products if c in test_products_by_customer]
    print(f"Evaluating hit rate@{TOP_N} on {len(eval_customers)} customers with purchases in both periods")

    hits = 0
    for customer_id in eval_customers:
        recs = recommend_for_customer(item_similarity, customer_products, descriptions, customer_id, n=TOP_N)
        recommended_codes = set(recs["stock_code"])
        actual_future_codes = test_products_by_customer[customer_id]
        if recommended_codes & actual_future_codes:
            hits += 1

    hit_rate = hits / len(eval_customers) if eval_customers else 0.0
    print(f"Hit Rate@{TOP_N}: {hit_rate:.3f} ({hits}/{len(eval_customers)})")
    return hit_rate


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

    print(f"\nRunning chronological hit rate evaluation ({EVAL_HOLDOUT_DAYS} day holdout)...")
    hit_rate = evaluate_hit_rate(df)
    if hit_rate is not None:
        append_metrics([["recommendation", "HitRate@5", round(hit_rate, 3)]])
        print("Saved metric to outputs/metrics/model_metrics.csv")

    print(f"\nBuilding final recommender on all data...")
    filtered = filter_interactions(df)
    print(f"Kept {len(filtered):,} interactions "
          f"({filtered['customer_id'].nunique():,} customers, {filtered['stock_code'].nunique():,} products) "
          f"after requiring >= {MIN_PRODUCTS_PER_CUSTOMER} products per customer and "
          f">= {MIN_CUSTOMERS_PER_PRODUCT} customers per product")

    item_similarity, customer_products = build_similarity_matrix(filtered)
    descriptions = filtered.groupby("stock_code")["description"].agg(lambda x: x.value_counts().index[0])

    # show a few examples: similar products for 3 popular items, and
    # recommendations for 3 customers
    example_products = filtered["stock_code"].value_counts().head(3).index
    print("\nExample: products similar to popular items")
    similar_rows = []
    for code in example_products:
        recs = recommend_similar_products(item_similarity, descriptions, code, n=TOP_N)
        recs.insert(0, "source_stock_code", code)
        recs.insert(1, "source_description", descriptions.get(code, ""))
        print(f"\n{code} - {descriptions.get(code, '')}")
        print(recs[["stock_code", "description", "similarity"]].to_string(index=False))
        similar_rows.append(recs)
    pd.concat(similar_rows, ignore_index=True).to_csv(
        os.path.join(TABLES_DIR, "recommendation_similar_products_examples.csv"), index=False
    )

    example_customers = pd.Series(list(customer_products.keys())).sample(n=3, random_state=RANDOM_STATE)
    print("\nExample: recommendations for individual customers")
    customer_rows = []
    for customer_id in example_customers:
        recs = recommend_for_customer(item_similarity, customer_products, descriptions, customer_id, n=TOP_N)
        recs.insert(0, "customer_id", customer_id)
        print(f"\nCustomer {customer_id}")
        print(recs[["stock_code", "description", "score"]].to_string(index=False))
        customer_rows.append(recs)
    pd.concat(customer_rows, ignore_index=True).to_csv(
        os.path.join(TABLES_DIR, "recommendation_customer_examples.csv"), index=False
    )

    print("\nSaved outputs/tables/recommendation_similar_products_examples.csv")
    print("Saved outputs/tables/recommendation_customer_examples.csv")


if __name__ == "__main__":
    main()
