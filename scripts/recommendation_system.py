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

The evaluation compares the item-item recommender against a simple
popularity baseline (recommend whatever sells the most) on the exact same
training period and the exact same evaluation customers, so the
comparison is fair.

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
DASHBOARD_PRODUCT_COUNT = 40

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


def recommend_popular(popularity_ranking, bought_codes, n=TOP_N):
    recs = [code for code in popularity_ranking if code not in bought_codes]
    return recs[:n]


def evaluate_hit_rate(df):
    cutoff = df["invoice_date"].max() - pd.Timedelta(days=EVAL_HOLDOUT_DAYS)
    train = df[df["invoice_date"] < cutoff]
    test = df[df["invoice_date"] >= cutoff]

    train = filter_interactions(train)
    if train.empty:
        print("Not enough training interactions to evaluate")
        return None, None

    item_similarity, customer_products = build_similarity_matrix(train)
    descriptions = train.groupby("stock_code")["description"].agg(lambda x: x.value_counts().index[0])

    # popularity baseline: most bought products in the same training period,
    # measured by number of distinct customers who bought each one
    popularity_ranking = (
        train.groupby("stock_code")["customer_id"].nunique().sort_values(ascending=False).index.tolist()
    )

    test_products_by_customer = test.groupby("customer_id")["stock_code"].apply(set).to_dict()

    eval_customers = [c for c in customer_products if c in test_products_by_customer]
    print(f"Evaluating hit rate@{TOP_N} on {len(eval_customers)} customers with purchases in both periods")

    item_item_hits = 0
    popularity_hits = 0
    for customer_id in eval_customers:
        actual_future_codes = test_products_by_customer[customer_id]

        recs = recommend_for_customer(item_similarity, customer_products, descriptions, customer_id, n=TOP_N)
        if set(recs["stock_code"]) & actual_future_codes:
            item_item_hits += 1

        popular_recs = recommend_popular(popularity_ranking, customer_products[customer_id], n=TOP_N)
        if set(popular_recs) & actual_future_codes:
            popularity_hits += 1

    n_eval = len(eval_customers)
    item_item_rate = item_item_hits / n_eval if n_eval else 0.0
    popularity_rate = popularity_hits / n_eval if n_eval else 0.0

    print(f"Popularity baseline   Hit Rate@{TOP_N}: {popularity_rate:.3f} ({popularity_hits}/{n_eval})")
    print(f"Item-item recommender Hit Rate@{TOP_N}: {item_item_rate:.3f} ({item_item_hits}/{n_eval})")
    print(f"Absolute improvement over popularity baseline: {item_item_rate - popularity_rate:.3f}")

    return popularity_rate, item_item_rate


def append_metrics(rows):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    new_rows = pd.DataFrame(rows, columns=["model", "metric", "value"])
    if os.path.exists(METRICS_PATH):
        existing = pd.read_csv(METRICS_PATH)
        existing = existing[~existing["model"].str.startswith("recommendation")]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(METRICS_PATH, index=False)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "purchases_clean.csv"), parse_dates=["invoice_date"])
    print(f"Full purchases table: {len(df):,} rows")

    print(f"\nRunning chronological hit rate evaluation ({EVAL_HOLDOUT_DAYS} day holdout)...")
    popularity_rate, item_item_rate = evaluate_hit_rate(df)
    if item_item_rate is not None:
        append_metrics([
            ["recommendation_popularity_baseline", "HitRate@5", round(popularity_rate, 3)],
            ["recommendation_item_item", "HitRate@5", round(item_item_rate, 3)],
        ])
        print("Saved metrics to outputs/metrics/model_metrics.csv")

    print(f"\nBuilding final recommender on all data...")
    filtered = filter_interactions(df)
    print(f"Kept {len(filtered):,} interactions "
          f"({filtered['customer_id'].nunique():,} customers, {filtered['stock_code'].nunique():,} products) "
          f"after requiring >= {MIN_PRODUCTS_PER_CUSTOMER} products per customer and "
          f">= {MIN_CUSTOMERS_PER_PRODUCT} customers per product")

    item_similarity, customer_products = build_similarity_matrix(filtered)
    descriptions = filtered.groupby("stock_code")["description"].agg(lambda x: x.value_counts().index[0])

    # build a similar-products table for the most popular products, this
    # backs both the printed examples and the dashboard product selector
    popular_products = filtered["stock_code"].value_counts().head(DASHBOARD_PRODUCT_COUNT).index
    print(f"\nBuilding similar-product recommendations for the {len(popular_products)} most popular products")
    similar_rows = []
    for code in popular_products:
        recs = recommend_similar_products(item_similarity, descriptions, code, n=TOP_N)
        recs.insert(0, "source_stock_code", code)
        recs.insert(1, "source_description", descriptions.get(code, ""))
        similar_rows.append(recs)
    similar_table = pd.concat(similar_rows, ignore_index=True)
    similar_table.to_csv(os.path.join(TABLES_DIR, "recommendation_similar_products_examples.csv"), index=False)
    print("Saved outputs/tables/recommendation_similar_products_examples.csv")

    print("\nExample: products similar to a few popular items")
    for code in popular_products[:3]:
        example = similar_table[similar_table["source_stock_code"] == code]
        print(f"\n{code} - {descriptions.get(code, '')}")
        print(example[["stock_code", "description", "similarity"]].to_string(index=False))

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
    print("\nSaved outputs/tables/recommendation_customer_examples.csv")


if __name__ == "__main__":
    main()
