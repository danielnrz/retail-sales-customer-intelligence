"""
Cleans the raw Online Retail II excel file and saves two processed datasets:

1. retail_clean.csv
   The full cleaned transaction log. Keeps cancelled orders and orders with
   no Customer ID, since those are still useful for business level analysis
   (revenue, cancellation rate, etc).

2. purchases_clean.csv
   A stricter subset used for the modeling scripts (segmentation, churn,
   demand prediction, price prediction, recommendations). Only contains
   completed purchases of real products from identified customers.

Run scripts/download_data.py first so data/raw/online_retail_II.xlsx exists.
"""

import os
import pandas as pd

RAW_FILE = os.path.join("data", "raw", "online_retail_II.xlsx")
PROCESSED_DIR = os.path.join("data", "processed")

# stock codes that showed up during inspection as postage, discounts, fees,
# manual adjustments, samples, etc, not real products
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "D", "S", "BANK CHARGES", "ADJUST",
    "AMAZONFEE", "DCGSSGIRL", "DCGSSBOY", "PADS", "CRUK", "B",
    "GIFT", "DCGSLBOY", "DCGSLGIRL",
}


def load_raw_data():
    print("Reading raw excel file, this can take a minute...")
    sheet_2009 = pd.read_excel(RAW_FILE, sheet_name="Year 2009-2010")
    sheet_2010 = pd.read_excel(RAW_FILE, sheet_name="Year 2010-2011")
    df = pd.concat([sheet_2009, sheet_2010], ignore_index=True)
    print(f"Loaded {len(df):,} raw rows")
    return df


def standardize_columns(df):
    df = df.rename(columns={
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "unit_price",
        "Customer ID": "customer_id",
        "Country": "country",
    })
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} exact duplicate rows")
    return df


def remove_bad_debt_rows(df):
    # a handful of rows record "Adjust bad debt" with a large negative price
    # and no customer, these are accounting entries, not product sales
    before = len(df)
    df = df[df["description"] != "Adjust bad debt"]
    print(f"Dropped {before - len(df):,} bad debt adjustment rows")
    return df


def fill_missing_descriptions(df):
    # build a lookup of the most common description used for each stock code
    lookup = (
        df.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda x: x.value_counts().index[0])
    )
    missing_before = df["description"].isna().sum()
    df["description"] = df["description"].fillna(df["stock_code"].map(lookup))
    df["description"] = df["description"].fillna("UNKNOWN ITEM")
    missing_after = df["description"].isna().sum()
    print(f"Filled {missing_before - missing_after:,} missing descriptions using stock code lookup")
    return df


def add_derived_columns(df):
    df["is_cancelled"] = df["invoice"].astype(str).str.startswith("C")
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["year"] = df["invoice_date"].dt.year
    df["month"] = df["invoice_date"].dt.month
    df["year_month"] = df["invoice_date"].dt.to_period("M").astype(str)
    df["week"] = df["invoice_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["invoice_date"].dt.day_name()
    df["hour"] = df["invoice_date"].dt.hour
    return df


def build_purchases_dataset(df):
    is_product = df["stock_code"].isin(NON_PRODUCT_CODES) == False
    purchases = df[
        (df["is_cancelled"] == False)
        & (df["quantity"] > 0)
        & (df["unit_price"] > 0)
        & (df["customer_id"].notna())
        & is_product
    ].copy()
    purchases["customer_id"] = purchases["customer_id"].astype(int)
    return purchases


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_raw_data()
    df = standardize_columns(df)
    df = remove_duplicates(df)
    df = remove_bad_debt_rows(df)
    df = fill_missing_descriptions(df)
    df = add_derived_columns(df)

    print()
    print(f"Full cleaned dataset: {len(df):,} rows")
    print(f"Cancelled orders: {df['is_cancelled'].sum():,}")
    print(f"Rows with no customer id: {df['customer_id'].isna().sum():,}")

    clean_path = os.path.join(PROCESSED_DIR, "retail_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"Saved {clean_path}")

    purchases = build_purchases_dataset(df)
    print()
    print(f"Purchases dataset (for modeling): {len(purchases):,} rows")
    print(f"Unique customers: {purchases['customer_id'].nunique():,}")
    print(f"Unique products: {purchases['stock_code'].nunique():,}")

    purchases_path = os.path.join(PROCESSED_DIR, "purchases_clean.csv")
    purchases.to_csv(purchases_path, index=False)
    print(f"Saved {purchases_path}")


if __name__ == "__main__":
    main()
