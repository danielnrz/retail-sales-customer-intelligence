"""
Builds a SQLite database from the cleaned retail transaction data.

Run scripts/clean_data.py first so data/processed/retail_clean.csv exists.
The database file itself is not tracked in git, it gets rebuilt from the
processed csv whenever it is needed.
"""

import os
import sqlite3
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
CLEAN_CSV = os.path.join(PROCESSED_DIR, "retail_clean.csv")
DB_PATH = os.path.join(PROCESSED_DIR, "retail.db")
SCHEMA_PATH = os.path.join("sql", "schema.sql")


def main():
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"{CLEAN_CSV} not found. Run scripts/clean_data.py first."
        )

    print("Loading cleaned transactions...")
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df):,} rows")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    with open(SCHEMA_PATH) as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    print("Created schema")

    df.to_sql("transactions", conn, if_exists="append", index=False)
    conn.commit()

    row_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"Loaded {row_count:,} rows into {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
