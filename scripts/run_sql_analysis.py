"""
Runs the business questions in sql/business_queries.sql against the SQLite
database and saves each result as a csv under outputs/tables/.

Run scripts/create_database.py first so data/processed/retail.db exists.
"""

import os
import re
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "processed", "retail.db")
QUERIES_PATH = os.path.join("sql", "business_queries.sql")
TABLES_DIR = os.path.join("outputs", "tables")


def parse_queries(sql_text):
    """splits business_queries.sql into a dict of {query_name: sql_string}
    using the '-- query: name' comment markers"""
    parts = re.split(r"--\s*query:\s*(\w+)", sql_text)
    # parts looks like: [preamble, name1, sql1, name2, sql2, ...]
    queries = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        sql = parts[i + 1].strip().rstrip(";")
        queries[name] = sql
    return queries


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found. Run scripts/create_database.py first.")

    os.makedirs(TABLES_DIR, exist_ok=True)

    with open(QUERIES_PATH) as f:
        sql_text = f.read()
    queries = parse_queries(sql_text)
    print(f"Found {len(queries)} queries in {QUERIES_PATH}")

    conn = sqlite3.connect(DB_PATH)

    for name, sql in queries.items():
        result = pd.read_sql_query(sql, conn)
        out_path = os.path.join(TABLES_DIR, f"{name}.csv")
        result.to_csv(out_path, index=False)
        print(f"\n{name} -> {out_path}")
        print(result.head(10).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
