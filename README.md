# Retail Sales Analytics and Customer Intelligence

This project analyzes about two years of transactions from a UK-based
online retailer and combines sales analysis, customer analysis,
forecasting, a few predictive models, and a simple recommendation system.
It uses Python for the data pipeline and modeling, and SQL for the
business-facing analysis, running against a SQLite database built from
the cleaned transactions.

## What I looked at

- How does revenue change over time, and is there seasonality?
- Which products and countries bring in the most revenue?
- Can recent sales history help forecast next week's total revenue?
- Can I estimate how many units of a specific product will sell next week?
- What kinds of customers show up in the transaction history, based on
  how recently, how often, and how much they buy?
- Which customers look like they are becoming inactive?
- Can transaction details help estimate what a product's price should be?
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
`-- notebooks/
    `-- retail_analysis.ipynb  walkthrough of the finished analysis
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
       `--> Recommendations
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

**Exploratory analysis.** `scripts/exploratory_analysis.py` produces 11
charts covering revenue over time, top products, country breakdown,
weekday/hour patterns, customer order frequency, customer spending
distribution, and cancellation rate over time.

## Models

| Task                 | Model                                   |
|----------------------|------------------------------------------|
| Sales forecasting    | Linear Regression (vs naive baseline)   |
| Demand prediction    | Gradient Boosting Regressor             |
| Segmentation         | KMeans (k=4)                            |
| Churn prediction     | Logistic Regression (vs Random Forest)  |
| Price estimation     | Ridge Regression                        |
| Recommendations      | Item-item cosine similarity             |

## Results

**Revenue overview.** GBP 20.47 million in revenue from 45,330 completed
orders, 5,881 active customers, average order value of GBP 451.47.
72.35% of customers placed more than one order. The cancellation rate
across all invoices is 15.46%.

**Sales forecasting.** Weekly revenue forecast, chronological train/test
split. The naive baseline (next week = this week) gets MAE 53,621. A
linear regression model using lag and rolling average features gets MAE
52,380. That is a real but modest improvement, weekly revenue for this
business is noisy and driven by a handful of large orders.

**Demand prediction.** Product-week level quantity prediction for the
top 50 selling products. Gradient boosting gets MAE 276 units, RMSE 520.
The model tracks the general demand level but misses some of the larger
spikes, which look like bulk or wholesale orders.

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
customers. On a 60-day chronological holdout, Hit Rate@5 is 0.21, meaning
about 1 in 5 customers had at least one of their top 5 recommended
products actually show up in their next two months of purchases.

A selection of the figures:

- `outputs/figures/01_monthly_revenue_trend.png`
- `outputs/figures/05_revenue_by_country.png`
- `outputs/figures/12_sales_forecast_actual_vs_predicted.png`
- `outputs/figures/15_customer_segments_scatter.png`
- `outputs/figures/18_churn_feature_importance.png`

All metrics referenced above come from `outputs/metrics/model_metrics.csv`,
which is produced by actually running the scripts in this repository.

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
- Simple models (linear regression, gradient boosting, logistic
  regression, Ridge, cosine similarity) all beat their respective
  baselines, though usually by a moderate margin rather than a dramatic
  one, which feels like an honest reflection of how noisy retail
  transaction data actually is.

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
the SQL analysis, and runs every model in order. It takes a couple of
minutes on a normal laptop. Individual scripts under `scripts/` can also
be run on their own once the earlier steps have produced their outputs.
