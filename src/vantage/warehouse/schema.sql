-- Vantage demo warehouse: a retail order-to-cash star-ish schema in SQLite.
-- Ten tables, 258,000 rows total, generated deterministically from a fixed seed.

PRAGMA foreign_keys = ON;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT    NOT NULL,
    department    TEXT    NOT NULL,
    is_perishable INTEGER NOT NULL
);

CREATE TABLE suppliers (
    supplier_id    INTEGER PRIMARY KEY,
    supplier_name  TEXT    NOT NULL,
    country        TEXT    NOT NULL,
    region         TEXT    NOT NULL,
    lead_time_days INTEGER NOT NULL,
    rating         REAL    NOT NULL
);

CREATE TABLE stores (
    store_id    INTEGER PRIMARY KEY,
    store_name  TEXT NOT NULL,
    city        TEXT NOT NULL,
    state       TEXT NOT NULL,
    country     TEXT NOT NULL,
    region      TEXT NOT NULL,
    opened_date TEXT NOT NULL,
    store_type  TEXT NOT NULL
);

CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    sku          TEXT    NOT NULL UNIQUE,
    product_name TEXT    NOT NULL,
    category_id  INTEGER NOT NULL REFERENCES categories(category_id),
    supplier_id  INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    unit_cost    REAL    NOT NULL,
    list_price   REAL    NOT NULL,
    launch_date  TEXT    NOT NULL,
    is_active    INTEGER NOT NULL
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    email        TEXT NOT NULL,
    signup_date  TEXT NOT NULL,
    country      TEXT NOT NULL,
    region       TEXT NOT NULL,
    city         TEXT NOT NULL,
    segment      TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL,
    is_active    INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    store_id         INTEGER NOT NULL REFERENCES stores(store_id),
    order_ts         TEXT    NOT NULL,
    order_date       TEXT    NOT NULL,
    channel          TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    currency         TEXT    NOT NULL,
    fulfillment_type TEXT    NOT NULL,
    promo_code       TEXT
);

CREATE TABLE order_items (
    order_item_id   INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL,
    unit_price      REAL    NOT NULL,
    discount_amount REAL    NOT NULL,
    line_total      REAL    NOT NULL
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    payment_ts TEXT    NOT NULL,
    method     TEXT    NOT NULL,
    amount     REAL    NOT NULL,
    status     TEXT    NOT NULL,
    processor  TEXT    NOT NULL
);

CREATE TABLE shipments (
    shipment_id         INTEGER PRIMARY KEY,
    order_id            INTEGER NOT NULL REFERENCES orders(order_id),
    carrier             TEXT    NOT NULL,
    shipped_ts          TEXT    NOT NULL,
    delivered_ts        TEXT,
    ship_cost           REAL    NOT NULL,
    status              TEXT    NOT NULL,
    destination_country TEXT    NOT NULL
);

CREATE TABLE returns (
    return_id     INTEGER PRIMARY KEY,
    order_item_id INTEGER NOT NULL REFERENCES order_items(order_item_id),
    return_ts     TEXT    NOT NULL,
    reason        TEXT    NOT NULL,
    quantity      INTEGER NOT NULL,
    refund_amount REAL    NOT NULL,
    condition     TEXT    NOT NULL
);

CREATE INDEX idx_products_category  ON products(category_id);
CREATE INDEX idx_products_supplier  ON products(supplier_id);
CREATE INDEX idx_orders_customer    ON orders(customer_id);
CREATE INDEX idx_orders_store       ON orders(store_id);
CREATE INDEX idx_orders_date        ON orders(order_date);
CREATE INDEX idx_order_items_order  ON order_items(order_id);
CREATE INDEX idx_order_items_prod   ON order_items(product_id);
CREATE INDEX idx_payments_order     ON payments(order_id);
CREATE INDEX idx_shipments_order    ON shipments(order_id);
CREATE INDEX idx_returns_item       ON returns(order_item_id);
