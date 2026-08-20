# Retail Sales Analytics and Customer Intelligence

Live dashboard: https://danielnrz.github.io/retail-sales-customer-intelligence/

This project analyzes about two years of transactions from a UK-based
online retailer and combines sales analysis, customer analysis,
forecasting, a few predictive models, and a simple recommendation system.
It uses Python for the data pipeline and modeling, and SQL for the
business-facing analysis, running against a SQLite database built from
the cleaned transactions. A static dashboard on top of the pipeline
outputs is linked above.

## What I looked at

- How does revenue change over time, and is there seasonality?
- Which products and countries bring in the most revenue?
- Can recent sales history help forecast next week's total revenue?
- Can I estimate how many units of a specific product will sell next week?
- What kinds of customers show up in the transaction history, based on
  how recently, how often, and how much they buy?
- Which customers look like they are becoming inactive?
- Can transaction details help estimate the unit price charged for a product?
- Given a customer's purchase history, what other products make sense to
  recommend?

## Dataset

- Online Retail II, UCI Machine Learning Repository, dataset ID 502
- https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.
  DOI: 10.24432/C5CG6D
- License: CC BY 4.0
- About 1.07 million transaction lines, December 2009 to December 2011
- Main columns: Invoice, StockCode, Description, Quantity, InvoiceDate,
  Price, Customer ID, Country

The raw data is downloaded by `scripts/download_data.py` and is not
stored in this repository. `data/raw/` and `data/processed/` are both
git-ignored, they get rebuilt by running the pipeline.

## Project structure

```text
.
|-- run_pipeline.py           runs the full pipeline end to end
|-- requirements.txt
|-- data/
|   |-- raw/                  downloaded dataset (git-ignored)
|   `-- processed/            cleaned csv files and the sqlite db (git-ignored)
|-- scripts/                  one script per pipeline step
|-- sql/                      schema and business queries
|-- outputs/
|   |-- figures/              charts from the analysis and models
|   |-- tables/                csv results from SQL queries and models
|   `-- metrics/               model_metrics.csv, all evaluation numbers
|-- docs/                     static dashboard, served with GitHub Pages
|   `-- data/                  small aggregated json files for the dashboard
|-- notebooks/
|   `-- retail_analysis.ipynb  walkthrough of the finished analysis
`-- tests/                    a few basic checks on the evaluation logic
```

## How the pipeline works

```text
Raw transactions
       |
       v
Data cleaning
       |
       v
SQLite database
       |
       +--> SQL analysis
       |
       +--> Sales forecasting
       |
       +--> Demand prediction
       |
       +--> Customer segmentation
       |
       +--> Churn prediction
       |
       +--> Price estimation
       |
       +--> Recommendations
       |
       `--> Dashboard data
```

## Analysis

**Cleaning.** The raw file has two sheets that get combined into
1,067,371 rows. After dropping 34,335 exact duplicates and 6 bad-debt
accounting rows, 1,033,030 rows remain. Cancelled orders (invoices
starting with "C") and orders with no Customer ID are kept in this full
dataset since they matter for business questions like cancellation rate.
A second, stricter dataset (`purchases_clean.csv`, 776,840 rows) is used
for anything customer or product level: completed orders only, real
products only (postage, discounts, manual adjustments and similar
non-product codes are excluded), positive quantity and price, and a
known Customer ID.

**SQL.** `sql/business_queries.sql` answers standard business questions
directly in SQL: total revenue, monthly revenue, best selling products,
revenue by country, top customers, repeat customer rate, cancellation
rate, and more. Results are saved as csv files under `outputs/tables/`.
Total revenue and the other completed-order queries filter out cancelled
invoices entirely rather than netting them off as negative revenue, the
cancellation rate itself is reported separately.

**Exploratory analysis.** `scripts/exploratory_analysis.py` produces 11
charts covering revenue over time, top products, country breakdown,
weekday/hour patterns, customer order frequency, customer spending
distribution, and cancellation rate over time.

**Evaluation method.** Sales forecasting and demand prediction both use a
three-way chronological split: training period, then a validation
period used only to pick which candidate model to use, then a final test
period that is touched exactly once, after the winning model is already
chosen. Earlier versions of this project picked the model directly on
the final test period, which quietly let the test result leak into model
selection. For demand prediction, the top-50 product list is also chosen
using only sales from before the test period, so the products being
modeled are not picked with knowledge of what happens to sell well later.

## Dashboard

The `docs/` folder is a small static site (plain HTML, CSS, and
JavaScript, charts drawn with Plotly.js from its CDN) published with
GitHub Pages. It has no backend and no build step, it just reads a
handful of small JSON files under `docs/data/`.

Those JSON files are generated by `scripts/build_dashboard_data.py` from
the same csv files in `outputs/tables/` and `outputs/metrics/` that the
rest of this README is based on, so the numbers on the dashboard match
the numbers reported here. Only aggregated results are published, no raw
transactions, no per-customer tables, and no database file.

## Models

| Task                 | Model                                       |
|-----------------------|---------------------------------------------|
| Sales forecasting     | Gradient Boosting Regressor (vs naive baseline) |
| Demand prediction     | Gradient Boosting Regressor (vs naive baseline) |
| Segmentation          | KMeans (k=4)                                |
| Churn prediction      | Logistic Regression (vs Random Forest)      |
| Price estimation      | Ridge Regression                            |
| Recommendations       | Item-item cosine similarity (vs popularity baseline) |

## Results

**Revenue overview.** GBP 20.47 million in revenue from 45,330 completed
orders, 5,881 active customers, average order value of GBP 451.47.
72.35% of customers placed more than one order. The cancellation rate
across all invoices is 15.46%.

**Sales forecasting.** Weekly revenue forecast, with the model chosen on
a validation period and scored once on a separate final test period. The
naive baseline (next week = this week) gets MAE 53,621 on those test
weeks. Gradient boosting, the model that won on validation, gets MAE
53,728, essentially tied with the baseline rather than clearly beating
it. Weekly revenue for this business is noisy and driven by a handful of
large orders, and a properly separated evaluation does not flatter the
model just because it looked good during model selection.

**Demand prediction.** Product-week level quantity prediction for the
top 50 selling products, with the product list chosen before the test
period and the model chosen on a validation period. The naive baseline
(next week = this week, per product) gets MAE 269.5 units. Gradient
boosting gets MAE 245.9 units, RMSE 418.4, a real improvement over the
baseline. The model tracks the general demand level but misses some of
the larger spikes, which look like bulk or wholesale orders.

**Customer segmentation.** RFM analysis with KMeans. Silhouette score
technically favors 2 clusters, but that just splits customers into low
and high value, so 4 clusters were used instead for a more useful
breakdown: High Value (recent, frequent, high spend), At Risk (used to
spend a fair amount, has gone quiet), Occasional (light but recent
buyers), and Lapsed (one or two small purchases a long time ago).

**Churn prediction.** Churn is defined as no purchase in the 90 days
after a cutoff date, using only pre-cutoff data for features. Logistic
regression gets F1 0.779 and ROC-AUC 0.812, slightly ahead of a random
forest (F1 0.761, ROC-AUC 0.793). Recency is the strongest predictor by
a wide margin.

**Price estimation.** Ridge regression on product description (TF-IDF),
quantity, country, and time features, trained on a reproducible sample
of 60,000 rows. MAE 1.31 versus a median-price baseline of MAE 1.82
(R2 0.287). Descriptions are closely tied to price since each product
usually sells within a narrow range, but bulk discounts and price
changes over time add noise a linear model cannot fully capture.

**Recommendations.** Item-item collaborative filtering using cosine
similarity on a customer-product matrix, restricted to customers with at
least 5 distinct products and products bought by at least 20 different
customers. On a 60-day chronological holdout, a plain popularity
baseline (recommend whatever sells the most, minus what the customer
already owns) gets Hit Rate@5 of 0.163 on the same evaluation customers.
The item-item recommender gets 0.210, an absolute improvement of about
0.047, so it is doing meaningfully better than just recommending
best-sellers.

A selection of the figures:

- `outputs/figures/01_monthly_revenue_trend.png`
- `outputs/figures/05_revenue_by_country.png`
- `outputs/figures/12_sales_forecast_actual_vs_predicted.png`
- `outputs/figures/15_customer_segments_scatter.png`
- `outputs/figures/18_churn_feature_importance.png`

All metrics referenced above come from `outputs/metrics/model_metrics.csv`,
which is produced by actually running the scripts in this repository, and
the same file feeds the live dashboard.

## Main findings

- Revenue is strongly seasonal, with a clear ramp-up into November each
  year ahead of the holiday period.
- The business is overwhelmingly UK based, other countries are a small
  share of total revenue even though 43 countries appear in the data.
- A small share of customers accounts for a disproportionate share of
  spend, which is why RFM features were log-transformed before
  clustering.
- Recency of last purchase is the single strongest signal for predicting
  future inactivity, stronger than total spend or order count.
- Demand prediction, churn, price estimation, and the recommender all
  beat their respective baselines by a real, if moderate, margin. Sales
  forecasting is the exception: once model selection is kept off the
  final test period, the winning model ends up roughly level with the
  naive baseline instead of clearly ahead of it, which says more about
  how hard short-horizon revenue forecasting is for this business than
  about the model itself.

## Limitations

- Churn here is an inactivity proxy built from purchase gaps, not a
  confirmed cancellation or contract end. There is no official churn
  label in this dataset.
- Price estimation predicts what price a transaction is likely to have,
  it is not a pricing strategy, elasticity, or profit optimization model.
- Demand and sales forecasts are based on historical patterns, they do
  not account for promotions, stockouts, or other business decisions
  that were not in the data.
- The recommendation system can only recommend from a customer's
  existing purchase pattern and from products with enough interaction
  history, new or rarely bought products are excluded by the thresholds
  used.
- The dataset represents a single UK-based online retailer selling
  mostly gift and homeware items, so these results are specific to this
  business and may not generalize elsewhere.

## Running the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

This downloads the dataset, cleans it, builds the SQLite database, runs
the SQL analysis, runs every model in order, and builds the dashboard
json files. It takes a couple of minutes on a normal laptop. Individual
scripts under `scripts/` can also be run on their own once the earlier
steps have produced their outputs.

To preview the dashboard locally after running the pipeline:

```bash
python -m http.server 8000 --directory docs
```

then open http://localhost:8000 in a browser.

A few basic checks on the evaluation logic and dashboard data live under
`tests/`:

```bash
python -m unittest tests/test_pipeline.py
```
