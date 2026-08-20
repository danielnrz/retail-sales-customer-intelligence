-- business questions answered directly in SQL against the transactions table
-- each query is labeled with a comment so run_sql_analysis.py can pull them
-- out one at a time and save the results

-- query: total_revenue
-- total net revenue across the whole dataset (cancellations included as
-- negative revenue since they reduce what the business actually kept)
SELECT ROUND(SUM(revenue), 2) AS total_revenue
FROM transactions
WHERE is_cancelled = 0;

-- query: order_count
-- number of distinct completed orders (invoices)
SELECT COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0;

-- query: active_customers
-- number of distinct customers who made at least one completed purchase
SELECT COUNT(DISTINCT customer_id) AS active_customers
FROM transactions
WHERE is_cancelled = 0 AND customer_id IS NOT NULL;

-- query: monthly_revenue
-- revenue and order count by calendar month
SELECT
    year_month,
    ROUND(SUM(revenue), 2) AS monthly_revenue,
    COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0
GROUP BY year_month
ORDER BY year_month;

-- query: average_order_value
-- average revenue per completed order
SELECT ROUND(SUM(revenue) / COUNT(DISTINCT invoice), 2) AS average_order_value
FROM transactions
WHERE is_cancelled = 0;

-- query: top_products_by_quantity
-- best selling products measured in units sold
SELECT
    stock_code,
    description,
    SUM(quantity) AS total_units_sold
FROM transactions
WHERE is_cancelled = 0 AND quantity > 0
GROUP BY stock_code, description
ORDER BY total_units_sold DESC
LIMIT 15;

-- query: top_products_by_revenue
-- products that generated the most revenue
SELECT
    stock_code,
    description,
    ROUND(SUM(revenue), 2) AS total_revenue
FROM transactions
WHERE is_cancelled = 0 AND quantity > 0
GROUP BY stock_code, description
ORDER BY total_revenue DESC
LIMIT 15;

-- query: revenue_by_country
-- revenue and order count broken down by country
SELECT
    country,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0
GROUP BY country
ORDER BY total_revenue DESC
LIMIT 20;

-- query: top_customers_by_revenue
-- highest spending identified customers
SELECT
    customer_id,
    country,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0 AND customer_id IS NOT NULL
GROUP BY customer_id
ORDER BY total_revenue DESC
LIMIT 20;

-- query: repeat_customer_rate
-- share of identified customers who placed more than one order
SELECT
    SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS repeat_customer_pct
FROM (
    SELECT customer_id, COUNT(DISTINCT invoice) AS order_count
    FROM transactions
    WHERE is_cancelled = 0 AND customer_id IS NOT NULL
    GROUP BY customer_id
);

-- query: cancellation_rate
-- share of all invoices that were cancelled
SELECT
    SUM(CASE WHEN is_cancelled = 1 THEN 1 ELSE 0 END) AS cancelled_invoices,
    COUNT(DISTINCT invoice) AS total_invoices,
    ROUND(100.0 * SUM(CASE WHEN is_cancelled = 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT invoice), 2) AS cancellation_pct
FROM (
    SELECT DISTINCT invoice, is_cancelled FROM transactions
);

-- query: revenue_by_weekday
-- revenue broken down by day of week
SELECT
    day_of_week,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0
GROUP BY day_of_week
ORDER BY total_revenue DESC;

-- query: revenue_by_hour
-- revenue broken down by hour of day
SELECT
    hour,
    ROUND(SUM(revenue), 2) AS total_revenue,
    COUNT(DISTINCT invoice) AS order_count
FROM transactions
WHERE is_cancelled = 0
GROUP BY hour
ORDER BY hour;

-- query: monthly_active_customers
-- number of distinct customers who purchased in each month
SELECT
    year_month,
    COUNT(DISTINCT customer_id) AS active_customers
FROM transactions
WHERE is_cancelled = 0 AND customer_id IS NOT NULL
GROUP BY year_month
ORDER BY year_month;
