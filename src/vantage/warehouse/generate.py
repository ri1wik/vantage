"""Deterministic generator for the Vantage demo warehouse.

Produces exactly 258,000 rows across ten tables from a fixed seed, so every
benchmark number in this repo is reproducible on any machine::

    python -m vantage.warehouse.generate --out data/warehouse.db
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20240917

# Row budget. The totals are fixed so `vantage-bench` gold answers stay stable.
ROW_PLAN = {
    "categories": 40,
    "suppliers": 200,
    "stores": 60,
    "products": 2_500,
    "customers": 12_000,
    "orders": 42_000,
    "order_items": 118_000,
    "payments": 44_000,
    "shipments": 34_000,
    "returns": 5_200,
}
TOTAL_ROWS = sum(ROW_PLAN.values())  # 258_000

WINDOW_START = date(2023, 1, 1)
WINDOW_END = date(2025, 12, 31)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days

DEPARTMENTS = ["Electronics", "Home", "Apparel", "Grocery", "Sports", "Beauty", "Toys", "Outdoors"]
CATEGORY_NOUNS = [
    "Headphones", "Laptops", "Monitors", "Cookware", "Bedding", "Lighting",
    "Jackets", "Footwear", "Denim", "Coffee", "Snacks", "Produce",
    "Cycling", "Running", "Yoga", "Skincare", "Haircare", "Fragrance",
    "Puzzles", "Figures", "Board Games", "Camping", "Hiking", "Fishing",
    "Tablets", "Wearables", "Speakers", "Storage", "Rugs", "Cutlery",
    "Sweaters", "Accessories", "Tea", "Dairy", "Weights", "Swim",
    "Makeup", "Bath", "Models", "Backpacks",
]
REGIONS = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe": ["United Kingdom", "Germany", "France", "Spain", "Netherlands"],
    "APAC": ["India", "Japan", "Australia", "Singapore"],
    "LATAM": ["Brazil", "Chile", "Colombia"],
}
CITIES = [
    "Austin", "Denver", "Seattle", "Toronto", "Monterrey", "London", "Berlin",
    "Lyon", "Madrid", "Utrecht", "Bengaluru", "Osaka", "Melbourne", "Singapore",
    "Sao Paulo", "Santiago", "Bogota", "Chicago", "Boston", "Dublin",
]
STATES = ["TX", "CO", "WA", "ON", "NL", "LDN", "BE", "ARA", "MD", "UT", "KA", "OS", "VIC", "SG", "SP", "RM", "DC", "IL", "MA", "D"]
SEGMENTS = ["Consumer", "SMB", "Enterprise"]
LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
CHANNELS = ["web", "mobile_app", "in_store", "phone", "marketplace"]
ORDER_STATUSES = ["completed", "completed", "completed", "completed", "cancelled", "pending", "refunded"]
FULFILLMENT = ["ship", "ship", "ship", "ship", "pickup", "digital"]
STORE_TYPES = ["flagship", "outlet", "popup", "warehouse"]
CURRENCIES = ["USD", "EUR", "GBP", "INR", "AUD"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "gift_card", "bank_transfer", "apple_pay"]
PAYMENT_STATUSES = ["captured", "captured", "captured", "captured", "authorized", "failed", "refunded"]
PROCESSORS = ["stripe", "adyen", "braintree", "worldpay"]
CARRIERS = ["UPS", "FedEx", "DHL", "USPS", "Royal Mail", "Blue Dart"]
SHIP_STATUSES = ["delivered", "delivered", "delivered", "in_transit", "lost", "returned_to_sender"]
RETURN_REASONS = ["damaged", "wrong_item", "no_longer_needed", "size_issue", "late_delivery", "quality"]
RETURN_CONDITIONS = ["resellable", "refurbish", "scrap"]
PROMO_CODES = [None, None, None, None, "SAVE10", "WELCOME15", "FREESHIP", "BUNDLE20", "VIP25"]
FIRST_NAMES = [
    "Ana", "Marcus", "Priya", "Kenji", "Sofia", "Liam", "Nadia", "Tomas", "Grace", "Omar",
    "Elena", "Noah", "Aisha", "Diego", "Mia", "Ravi", "Clara", "Yusuf", "Freya", "Hugo",
]
LAST_NAMES = [
    "Okafor", "Lindqvist", "Sharma", "Tanaka", "Duarte", "Byrne", "Haddad", "Novak", "Bell", "Rahman",
    "Petrova", "Wallace", "Diallo", "Moreno", "Chen", "Iyer", "Fontaine", "Yilmaz", "Berg", "Costa",
]
ADJECTIVES = ["Compact", "Premium", "Everyday", "Pro", "Lite", "Trail", "Urban", "Studio", "Classic", "Nordic"]
NOUNS = ["Kit", "Set", "Pack", "Edition", "Series", "Bundle", "Model", "Line", "Collection", "Unit"]


def _day(rng: random.Random) -> date:
    """A date in-window, weighted so later months carry more volume."""
    u = rng.random() ** 0.8  # skew toward recent
    return WINDOW_START + timedelta(days=int(u * WINDOW_DAYS))


def _ts(d: date, rng: random.Random) -> str:
    return datetime(d.year, d.month, d.day, rng.randrange(24), rng.randrange(60), rng.randrange(60)).isoformat(sep=" ")


def _country_region(rng: random.Random) -> tuple[str, str]:
    region = rng.choice(list(REGIONS))
    return region, rng.choice(REGIONS[region])


def build(db_path: Path, seed: int = SEED) -> dict[str, int]:
    """(Re)build the warehouse at ``db_path``. Returns the per-table row counts."""
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    # --- categories -------------------------------------------------------
    categories = []
    for i in range(ROW_PLAN["categories"]):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        categories.append((i + 1, CATEGORY_NOUNS[i], dept, 1 if dept == "Grocery" else 0))
    conn.executemany("INSERT INTO categories VALUES (?,?,?,?)", categories)

    # --- suppliers --------------------------------------------------------
    suppliers = []
    for i in range(ROW_PLAN["suppliers"]):
        region, country = _country_region(rng)
        suppliers.append((
            i + 1,
            f"{rng.choice(ADJECTIVES)} {rng.choice(LAST_NAMES)} Supply Co",
            country, region,
            rng.randint(2, 45),
            round(rng.uniform(2.5, 5.0), 2),
        ))
    conn.executemany("INSERT INTO suppliers VALUES (?,?,?,?,?,?)", suppliers)

    # --- stores -----------------------------------------------------------
    stores = []
    for i in range(ROW_PLAN["stores"]):
        region, country = _country_region(rng)
        idx = rng.randrange(len(CITIES))
        stores.append((
            i + 1,
            f"{CITIES[idx]} {STORE_TYPES[i % len(STORE_TYPES)].title()} #{i + 1}",
            CITIES[idx], STATES[idx], country, region,
            (WINDOW_START - timedelta(days=rng.randint(200, 3000))).isoformat(),
            STORE_TYPES[i % len(STORE_TYPES)],
        ))
    conn.executemany("INSERT INTO stores VALUES (?,?,?,?,?,?,?,?)", stores)

    # --- products ---------------------------------------------------------
    products = []
    for i in range(ROW_PLAN["products"]):
        cat = rng.randint(1, ROW_PLAN["categories"])
        cost = round(rng.uniform(3.0, 480.0), 2)
        products.append((
            i + 1,
            f"SKU-{i + 1:06d}",
            f"{rng.choice(ADJECTIVES)} {CATEGORY_NOUNS[cat - 1]} {rng.choice(NOUNS)}",
            cat,
            rng.randint(1, ROW_PLAN["suppliers"]),
            cost,
            round(cost * rng.uniform(1.25, 2.6), 2),
            (WINDOW_START - timedelta(days=rng.randint(0, 1200))).isoformat(),
            1 if rng.random() > 0.12 else 0,
        ))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", products)
    price_of = {p[0]: p[6] for p in products}

    # --- customers --------------------------------------------------------
    customers = []
    for i in range(ROW_PLAN["customers"]):
        region, country = _country_region(rng)
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        customers.append((
            i + 1, fn, ln,
            f"{fn.lower()}.{ln.lower()}{i + 1}@example.com",
            (WINDOW_START - timedelta(days=rng.randint(0, 1600))).isoformat(),
            country, region, rng.choice(CITIES),
            rng.choice(SEGMENTS),
            rng.choices(LOYALTY_TIERS, weights=[45, 30, 18, 7])[0],
            1 if rng.random() > 0.09 else 0,
        ))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", customers)

    # --- orders -----------------------------------------------------------
    orders, order_days = [], {}
    for i in range(ROW_PLAN["orders"]):
        d = _day(rng)
        order_days[i + 1] = d
        orders.append((
            i + 1,
            rng.randint(1, ROW_PLAN["customers"]),
            rng.randint(1, ROW_PLAN["stores"]),
            _ts(d, rng), d.isoformat(),
            rng.choices(CHANNELS, weights=[38, 27, 18, 7, 10])[0],
            rng.choice(ORDER_STATUSES),
            rng.choices(CURRENCIES, weights=[52, 20, 12, 10, 6])[0],
            rng.choice(FULFILLMENT),
            rng.choice(PROMO_CODES),
        ))
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", orders)

    # --- order_items ------------------------------------------------------
    # Every order gets at least one line; the remainder are spread randomly.
    n_orders, n_items = ROW_PLAN["orders"], ROW_PLAN["order_items"]
    item_order_ids = list(range(1, n_orders + 1))
    item_order_ids += [rng.randint(1, n_orders) for _ in range(n_items - n_orders)]
    item_order_ids.sort()

    items = []
    for idx, oid in enumerate(item_order_ids, start=1):
        pid = rng.randint(1, ROW_PLAN["products"])
        qty = rng.choices([1, 2, 3, 4, 6, 10], weights=[52, 24, 12, 6, 4, 2])[0]
        unit = round(price_of[pid] * rng.uniform(0.9, 1.08), 2)
        disc = round(unit * qty * rng.choice([0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.25]), 2)
        items.append((idx, oid, pid, qty, unit, disc, round(unit * qty - disc, 2)))
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)", items)

    # --- payments ---------------------------------------------------------
    order_total: dict[int, float] = {}
    for _, oid, _, _, _, _, line_total in items:
        order_total[oid] = order_total.get(oid, 0.0) + line_total

    pay_order_ids = list(range(1, n_orders + 1))
    pay_order_ids += [rng.randint(1, n_orders) for _ in range(ROW_PLAN["payments"] - n_orders)]
    pay_order_ids.sort()
    payments = []
    for idx, oid in enumerate(pay_order_ids, start=1):
        d = order_days[oid] + timedelta(days=rng.randint(0, 2))
        payments.append((
            idx, oid, _ts(d, rng),
            rng.choices(PAYMENT_METHODS, weights=[40, 18, 16, 6, 10, 10])[0],
            round(order_total.get(oid, 0.0) * rng.uniform(0.45, 1.0), 2),
            rng.choice(PAYMENT_STATUSES),
            rng.choice(PROCESSORS),
        ))
    conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?,?,?)", payments)

    # --- shipments --------------------------------------------------------
    ship_order_ids = rng.sample(range(1, n_orders + 1), ROW_PLAN["shipments"])
    ship_order_ids.sort()
    shipments = []
    for idx, oid in enumerate(ship_order_ids, start=1):
        shipped = order_days[oid] + timedelta(days=rng.randint(0, 4))
        status = rng.choices(SHIP_STATUSES, weights=[40, 30, 18, 7, 2, 3])[0]
        delivered = (shipped + timedelta(days=rng.randint(1, 14))).isoformat() if status == "delivered" else None
        region, country = _country_region(rng)
        shipments.append((
            idx, oid, rng.choice(CARRIERS), shipped.isoformat(), delivered,
            round(rng.uniform(2.5, 38.0), 2), status, country,
        ))
    conn.executemany("INSERT INTO shipments VALUES (?,?,?,?,?,?,?,?)", shipments)

    # --- returns ----------------------------------------------------------
    return_item_ids = rng.sample(range(1, len(items) + 1), ROW_PLAN["returns"])
    return_item_ids.sort()
    returns = []
    for idx, iid in enumerate(return_item_ids, start=1):
        _, oid, _, qty, unit, _, line_total = items[iid - 1]
        rq = rng.randint(1, qty)
        d = order_days[oid] + timedelta(days=rng.randint(3, 60))
        returns.append((
            idx, iid, _ts(d, rng),
            rng.choice(RETURN_REASONS), rq,
            round(min(line_total, unit * rq), 2),
            rng.choice(RETURN_CONDITIONS),
        ))
    conn.executemany("INSERT INTO returns VALUES (?,?,?,?,?,?,?)", returns)

    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ROW_PLAN}
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Vantage demo warehouse.")
    ap.add_argument("--out", default="data/warehouse.db", help="output SQLite path")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    counts = build(Path(args.out).resolve(), seed=args.seed)
    total = sum(counts.values())
    width = max(len(t) for t in counts)
    for table, n in counts.items():
        print(f"  {table.ljust(width)}  {n:>8,}")
    print(f"  {'TOTAL'.ljust(width)}  {total:>8,}")
    return 0 if total == TOTAL_ROWS else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
