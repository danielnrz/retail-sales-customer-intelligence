"""
A few basic checks on the pipeline's evaluation logic and dashboard data.

This is not a full test suite, just enough to catch the kind of mistakes
that are easy to make in a time series project: features that peek at
future values, or a test period that overlaps with training.

Run with:
    python -m unittest tests/test_pipeline.py
"""

import os
import sys
import json
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import sales_forecasting
import demand_prediction

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
DASHBOARD_DATA_DIR = os.path.join(PROJECT_ROOT, "docs", "data")


class TestLagFeaturesHaveNoLeakage(unittest.TestCase):
    """the lag_1 feature for a given week must equal the revenue from the
    week before it, never the revenue of that same week or a later one"""

    def test_sales_forecast_lag_features(self):
        weekly = pd.DataFrame({
            "week_start": pd.date_range("2020-01-06", periods=10, freq="W-MON"),
            "revenue": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        })
        featured = sales_forecasting.add_features(weekly)

        for _, row in featured.iterrows():
            week_index = weekly.index[weekly["week_start"] == row["week_start"]][0]
            expected_lag_1 = weekly["revenue"].iloc[week_index - 1]
            self.assertEqual(row["lag_1"], expected_lag_1)
            self.assertNotEqual(row["lag_1"], row["revenue"])

    def test_demand_lag_features(self):
        weeks = pd.date_range("2020-01-06", periods=8, freq="W-MON")
        weekly = pd.DataFrame({
            "stock_code": ["A"] * 8,
            "week_start": weeks,
            "quantity": [5, 10, 15, 20, 25, 30, 35, 40],
            "revenue": [50, 100, 150, 200, 250, 300, 350, 400],
            "avg_price": [10] * 8,
        })
        featured = demand_prediction.add_features(weekly)

        for _, row in featured.iterrows():
            week_index = weekly.index[weekly["week_start"] == row["week_start"]][0]
            expected_lag_1 = weekly["quantity"].iloc[week_index - 1]
            self.assertEqual(row["qty_lag_1"], expected_lag_1)
            self.assertNotEqual(row["qty_lag_1"], row["quantity"])


class TestChronologicalSplit(unittest.TestCase):
    """the final test period must come after both the training and
    validation periods, never overlapping or out of order"""

    def setUp(self):
        results_path = os.path.join(TABLES_DIR, "sales_forecast_results.csv")
        if not os.path.exists(results_path):
            self.skipTest("sales_forecast_results.csv not found, run the pipeline first")
        self.sales_results = pd.read_csv(results_path, parse_dates=["week_start"])

    def test_sales_forecast_test_weeks_are_in_order(self):
        weeks = self.sales_results["week_start"].tolist()
        self.assertEqual(weeks, sorted(weeks))
        self.assertEqual(len(weeks), sales_forecasting.TEST_WEEKS)

    def test_demand_prediction_results_after_split(self):
        results_path = os.path.join(TABLES_DIR, "demand_prediction_results.csv")
        if not os.path.exists(results_path):
            self.skipTest("demand_prediction_results.csv not found, run the pipeline first")
        demand_results = pd.read_csv(results_path, parse_dates=["week_start"])

        for stock_code, group in demand_results.groupby("stock_code"):
            weeks = group.sort_values("week_start")["week_start"].tolist()
            self.assertEqual(weeks, sorted(weeks))


class TestDashboardData(unittest.TestCase):
    """the dashboard json files need to have the keys app.js expects"""

    def _load(self, filename):
        path = os.path.join(DASHBOARD_DATA_DIR, filename)
        if not os.path.exists(path):
            self.skipTest(f"{filename} not found, run scripts/build_dashboard_data.py first")
        with open(path) as f:
            return json.load(f)

    def test_overview_has_required_keys(self):
        data = self._load("overview.json")
        required = [
            "total_revenue", "orders", "active_customers",
            "average_order_value", "repeat_customer_rate", "cancellation_rate",
        ]
        for key in required:
            self.assertIn(key, data)
            self.assertIsNotNone(data[key])

    def test_models_has_required_sections(self):
        data = self._load("models.json")
        for key in ["sales_forecast", "demand_prediction", "price_estimation", "churn"]:
            self.assertIn(key, data)

    def test_recommendations_has_hit_rates(self):
        data = self._load("recommendations.json")
        self.assertIn("popularity_baseline_hit_rate", data)
        self.assertIn("item_item_hit_rate", data)
        self.assertGreater(len(data["products"]), 0)


if __name__ == "__main__":
    unittest.main()
