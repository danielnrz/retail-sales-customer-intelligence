-- schema for the retail sales SQLite database
-- one table holding the cleaned transaction log, this is enough for the
-- kind of business questions this project asks

DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    invoice       TEXT,
    stock_code    TEXT,
    description   TEXT,
    quantity      INTEGER,
    invoice_date  TEXT,
    unit_price    REAL,
    customer_id   INTEGER,
    country       TEXT,
    is_cancelled  INTEGER,
    revenue       REAL,
    year          INTEGER,
    month         INTEGER,
    year_month    TEXT,
    week          INTEGER,
    day_of_week   TEXT,
    hour          INTEGER
);

CREATE INDEX idx_transactions_invoice_date ON transactions (invoice_date);
CREATE INDEX idx_transactions_customer_id ON transactions (customer_id);
CREATE INDEX idx_transactions_stock_code ON transactions (stock_code);
CREATE INDEX idx_transactions_invoice ON transactions (invoice);
