"""
Exploratory analysis of the cleaned retail transactions.

Looks at how sales change over time, which products and countries matter
most, and how customers behave. Saves one png per question under
outputs/figures/.

Run scripts/clean_data.py first so data/processed/retail_clean.csv exists.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_data():
    df = pd.read_csv(PROCESSED_DIR + "/retail_clean.csv", parse_dates=["invoice_date"])
    purchases = pd.read_csv(PROCESSED_DIR + "/purchases_clean.csv", parse_dates=["invoice_date"])
    return df, purchases


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def plot_monthly_revenue(df):
    completed = df[df["is_cancelled"] == False]
    monthly = completed.groupby("year_month")["revenue"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_title("Monthly Revenue Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (GBP)")
    ax.tick_params(axis="x", rotation=90)
    save_fig(fig, "01_monthly_revenue_trend.png")


def plot_weekly_orders(df):
    completed = df[df["is_cancelled"] == False].copy()
    completed["week_start"] = completed["invoice_date"].dt.to_period("W").dt.start_time
    weekly_orders = completed.groupby("week_start")["invoice"].nunique()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekly_orders.index, weekly_orders.values)
    ax.set_title("Weekly Order Volume Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Number of Orders")
    save_fig(fig, "02_weekly_order_volume.png")


def plot_top_products_revenue(purchases):
    top = purchases.groupby("description")["revenue"].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue (GBP)")
    save_fig(fig, "03_top_products_by_revenue.png")


def plot_top_products_quantity(purchases):
    top = purchases.groupby("description")["quantity"].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title("Top 10 Products by Units Sold")
    ax.set_xlabel("Units Sold")
    save_fig(fig, "04_top_products_by_quantity.png")


def plot_revenue_by_country(df):
    completed = df[df["is_cancelled"] == False]
    top = completed.groupby("country")["revenue"].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title("Top 10 Countries by Revenue")
    ax.set_xlabel("Revenue (GBP)")
    save_fig(fig, "05_revenue_by_country.png")


def plot_revenue_by_weekday(df):
    completed = df[df["is_cancelled"] == False]
    by_day = completed.groupby("day_of_week")["revenue"].sum().reindex(WEEKDAY_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_day.index, by_day.values)
    ax.set_title("Revenue by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue (GBP)")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "06_revenue_by_weekday.png")


def plot_revenue_by_hour(df):
    completed = df[df["is_cancelled"] == False]
    by_hour = completed.groupby("hour")["revenue"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_hour.index, by_hour.values)
    ax.set_title("Revenue by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Revenue (GBP)")
    save_fig(fig, "07_revenue_by_hour.png")


def plot_customer_order_frequency(purchases):
    orders_per_customer = purchases.groupby("customer_id")["invoice"].nunique()
    capped = orders_per_customer.clip(upper=20)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(capped, bins=20, edgecolor="white")
    ax.set_title("Customer Order Frequency (capped at 20 orders)")
    ax.set_xlabel("Number of Orders")
    ax.set_ylabel("Number of Customers")
    save_fig(fig, "08_customer_order_frequency.png")


def plot_customer_spending_distribution(purchases):
    spend_per_customer = purchases.groupby("customer_id")["revenue"].sum()
    spend_per_customer = spend_per_customer[spend_per_customer > 0]

    # use log-spaced bins since spending is heavily right skewed
    log_bins = np.logspace(np.log10(spend_per_customer.min()), np.log10(spend_per_customer.max()), 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(spend_per_customer, bins=log_bins, edgecolor="white")
    ax.set_xscale("log")
    ax.set_title("Customer Total Spending Distribution")
    ax.set_xlabel("Total Spend (GBP, log scale)")
    ax.set_ylabel("Number of Customers")
    save_fig(fig, "09_customer_spending_distribution.png")


def plot_cancellation_rate_by_month(df):
    invoice_level = df.drop_duplicates(subset="invoice")[["invoice", "year_month", "is_cancelled"]]
    rate = invoice_level.groupby("year_month")["is_cancelled"].mean().sort_index() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rate.index, rate.values, marker="o", color="firebrick")
    ax.set_title("Cancellation Rate by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cancelled Invoices (%)")
    ax.tick_params(axis="x", rotation=90)
    save_fig(fig, "10_cancellation_rate_by_month.png")


def plot_average_order_value_trend(df):
    completed = df[df["is_cancelled"] == False]
    monthly_revenue = completed.groupby("year_month")["revenue"].sum()
    monthly_orders = completed.groupby("year_month")["invoice"].nunique()
    aov = (monthly_revenue / monthly_orders).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aov.index, aov.values, marker="o", color="darkgreen")
    ax.set_title("Average Order Value by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Order Value (GBP)")
    ax.tick_params(axis="x", rotation=90)
    save_fig(fig, "11_average_order_value_trend.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading processed data...")
    df, purchases = load_data()

    plot_monthly_revenue(df)
    plot_weekly_orders(df)
    plot_top_products_revenue(purchases)
    plot_top_products_quantity(purchases)
    plot_revenue_by_country(df)
    plot_revenue_by_weekday(df)
    plot_revenue_by_hour(df)
    plot_customer_order_frequency(purchases)
    plot_customer_spending_distribution(purchases)
    plot_cancellation_rate_by_month(df)
    plot_average_order_value_trend(df)

    print("\nDone. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
