"""
Runs the full retail analytics pipeline from start to finish:

download -> clean -> database -> SQL analysis -> EDA -> sales forecasting
-> demand prediction -> segmentation -> churn prediction -> price
estimation -> recommendations -> dashboard data

Just run:
    python run_pipeline.py

Each step is a normal script under scripts/, this file just calls them in
the right order and stops if one of them fails.
"""

import subprocess
import sys
import time

PIPELINE_STEPS = [
    ("scripts/download_data.py", "Downloading raw dataset"),
    ("scripts/clean_data.py", "Cleaning data"),
    ("scripts/create_database.py", "Building SQLite database"),
    ("scripts/run_sql_analysis.py", "Running SQL business queries"),
    ("scripts/exploratory_analysis.py", "Running exploratory analysis"),
    ("scripts/sales_forecasting.py", "Forecasting weekly sales"),
    ("scripts/demand_prediction.py", "Predicting product demand"),
    ("scripts/customer_segmentation.py", "Segmenting customers"),
    ("scripts/churn_prediction.py", "Predicting churn"),
    ("scripts/price_prediction.py", "Estimating product prices"),
    ("scripts/recommendation_system.py", "Building recommendation system"),
    ("scripts/build_dashboard_data.py", "Building dashboard data"),
]


def run_step(script_path, description):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)

    start = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n{script_path} failed after {elapsed:.1f} seconds")
        sys.exit(1)

    print(f"\n{script_path} finished in {elapsed:.1f} seconds")


def main():
    pipeline_start = time.time()

    for script_path, description in PIPELINE_STEPS:
        run_step(script_path, description)

    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print(f"Pipeline finished in {total_elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
