"""The deterministic baseline analyst.

``vantage-bench`` needs a control that runs in CI with no API key, no network and
no sampling variance, and it has to run through the *same* graph, guardrails and
scorer as a hosted model. This module is that control: a rule-based reader that
turns a question into the same :class:`~vantage.plan.Plan` JSON a hosted model is
asked for, then compiles it with the shared :class:`~vantage.sql_compiler.SqlCompiler`.

It is a baseline, not a ceiling. It understands the question grammar this
warehouse actually gets asked (a measure, some dimensions, filters, a time grain,
an ordering) and refuses everything else, which is exactly the behaviour a
benchmark control should have: it scores well where the grammar covers the
question and visibly fails where it does not, so a hosted model's score means
something relative to it.

``fault_profile`` injects one specific, realistic first-attempt error so the
self-correction tier can be measured deterministically. The injected model
repairs itself only when the critic hands back a real diagnosis, so the tier
scores the repair loop rather than the retry count.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..plan import Dimension, Filter, Measure, Plan, Refusal
from ..sql_compiler import CompileError, SqlCompiler
from ..warehouse.catalog import Catalog
from .base import LLMRequest, LLMResponse

# --------------------------------------------------------------------------
# Refusal rules. Ordered: the first match decides the category.
# --------------------------------------------------------------------------
WRITE_INTENT = re.compile(
    r"\b(delete|truncate|insert\s+into|wipe|purge|erase|overwrite|rollback)\b"
    # "drop" is a DDL verb and an analytics noun. Only the DDL sense counts:
    # "why did revenue drop in March" is a question, not a request to drop a table.
    r"|\bdrop\s+(the\s+)?\w*\s*(table|index|view|trigger|database|schema|column)\b"
    r"|\bupdate\s+\w+\s+set\b"
    r"|\bset\s+\w+\s*="
    # "update X to Y" / "set the tier of customer 12 to Platinum". The lookahead
    # keeps "change in revenue compared to 2024" out: there the verb is a noun.
    r"|\b(set|change|modify|update|reset|edit|rename)\s+"
    r"(?!in\b|of\b|for\b|from\b|by\b|over\b|between\b|since\b|versus\b|vs\b)"
    r"(the\s+)?[\w\s'\-]{0,60}?\bto\s+\S"
    r"|\bmark\s+[\w\s]{1,30}\s+as\b"
    r"|\b(add|create)\s+a?\s*(new\s+)?(row|record|order|customer|table|column)\b",
    re.IGNORECASE,
)

PII_INTENT = re.compile(
    r"\b(e-?mail(\s+address(es)?)?|phone|mobile\s+number|home\s+address|postal\s+address|"
    r"full\s+names?|first\s+and\s+last\s+names?|contact\s+details?|personally\s+identifiable|pii)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE = re.compile(
    r"\b(page\s*views?|pageviews?|web\s*sessions?|clicks?|click-?through|impressions?|"
    r"ad\s+spend|marketing\s+(spend|budget)|advertis\w+|seo|"
    r"employees?|staff|headcount|salar(y|ies)|payroll|hiring|"
    r"competitors?|market\s+share|"
    r"weather|holiday\s+calendar|"
    r"nps|csat|surveys?|customer\s+reviews?|star\s+ratings?|testimonials?|"
    r"inventory|stock\s+(level|on\s+hand|out)|stockouts?|warehouse\s+capacity|"
    r"social\s+media|newsletters?|open\s+rate|support\s+tickets?|helpdesk)\b",
    re.IGNORECASE,
)

UNSUPPORTED = re.compile(
    r"\b(predict|forecast|projection|project(ed)?\s+(revenue|sales|demand)|extrapolat\w+|"
    r"next\s+(quarter|month|year|week)|will\s+(we|it|sales|revenue)|"
    r"should\s+(we|i)|recommend|advise|"
    r"why\s+(did|is|are|has|have|does)|what\s+caused|root\s+cause|because\s+of\s+what)\b",
    re.IGNORECASE,
)

VAGUE = re.compile(
    r"^\s*(how\s+are\s+we\s+doing|how(?:'s| is)\s+business|show\s+me\s+everything|"
    r"give\s+me\s+the\s+data|tell\s+me\s+about\s+the\s+data|what(?:'s| is)\s+interesting|"
    r"anything\s+useful|summar(y|ise|ize)\s+everything)\b",
    re.IGNORECASE,
)


@dataclass
class MeasureRule:
    pattern: re.Pattern[str]
    build: Callable[[], Measure]


def _m(agg: str, table: str, column: str, alias: str, **kw: Any) -> Callable[[], Measure]:
    return lambda: Measure(agg=agg, table=table, column=column, alias=alias, **kw)


# Ordered most specific first: "how many refunds" must beat "refund".
MEASURE_RULES: list[MeasureRule] = [
    MeasureRule(
        re.compile(r"\breturn\s+rate\b|\brate\s+of\s+returns?\b", re.I),
        _m("ratio", "order_items", "quantity", "return_rate",
           numerator="SUM(COALESCE({returns}.quantity, 0))",
           denominator="SUM({order_items}.quantity)",
           extra_tables=["returns"]),
    ),
    MeasureRule(
        re.compile(r"\b(average\s+order\s+value|aov|basket\s+size)\b", re.I),
        _m("ratio", "order_items", "line_total", "average_order_value",
           numerator="SUM({order_items}.line_total)",
           denominator="COUNT(DISTINCT {orders}.order_id)",
           extra_tables=["orders"]),
    ),
    MeasureRule(
        re.compile(r"\b(payment\s+)?(failure|decline)\s+rate\b", re.I),
        _m("ratio", "payments", "payment_id", "failure_rate",
           numerator="SUM(CASE WHEN {payments}.status = 'failed' THEN 1 ELSE 0 END)",
           denominator="COUNT(*)"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+(returns?|refunds?|rmas?)\b|\breturn\s+count\b", re.I),
        _m("count", "returns", "return_id", "return_count"),
    ),
    MeasureRule(
        re.compile(r"\b(returned\s+units|units\s+returned|return(ed)?\s+quantity)\b", re.I),
        _m("sum", "returns", "quantity", "returned_units"),
    ),
    MeasureRule(
        re.compile(r"\b(refunds?|refunded|money\s+back|credited\s+back)\b", re.I),
        _m("sum", "returns", "refund_amount", "refund_amount"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+orders?\b|\border\s+count\b|"
                   r"\borders?\s+(were\s+)?placed\b|\bhow\s+many\s+.{0,20}orders?\b", re.I),
        _m("count_distinct", "orders", "order_id", "order_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+customers?\b|\bcustomer\s+count\b", re.I),
        _m("count_distinct", "customers", "customer_id", "customer_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+(products?|skus?|items?)\b|\bproduct\s+count\b", re.I),
        _m("count", "products", "product_id", "product_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+(stores?|shops?|locations?)\b", re.I),
        _m("count", "stores", "store_id", "store_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+(shipments?|parcels?|deliveries)\b", re.I),
        _m("count", "shipments", "shipment_id", "shipment_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+(payments?|charges?|transactions?)\b", re.I),
        _m("count", "payments", "payment_id", "payment_count"),
    ),
    MeasureRule(
        re.compile(r"\b(how\s+many|number\s+of|count\s+of)\s+suppliers?\b", re.I),
        _m("count", "suppliers", "supplier_id", "supplier_count"),
    ),
    MeasureRule(
        re.compile(r"\b(delivery\s+(time|days|duration)|days?\s+to\s+deliver|transit\s+time|"
                   r"how\s+long\s+.{0,20}deliver)\b", re.I),
        _m("expr", "shipments", "delivered_ts", "avg_delivery_days",
           expr="ROUND(AVG(julianday({shipments}.delivered_ts) - julianday({shipments}.shipped_ts)), 2)"),
    ),
    MeasureRule(
        re.compile(r"\b(shipping|freight|ship)\s+cost\b", re.I),
        _m("sum", "shipments", "ship_cost", "shipping_cost"),
    ),
    MeasureRule(re.compile(r"\blead\s+time\b", re.I), _m("avg", "suppliers", "lead_time_days", "avg_lead_time_days")),
    MeasureRule(
        re.compile(r"\b(supplier|quality|vendor)\s+ratings?\b|\bratings?\b", re.I),
        _m("avg", "suppliers", "rating", "avg_rating"),
    ),
    MeasureRule(
        re.compile(r"\b(gross\s+)?(margin|profit)\b", re.I),
        _m("expr", "order_items", "line_total", "gross_margin",
           expr="ROUND(SUM({order_items}.line_total - {products}.unit_cost * {order_items}.quantity), 2)",
           extra_tables=["products"]),
    ),
    MeasureRule(
        re.compile(r"\b(amount\s+(paid|tendered)|payment\s+amount|settled\s+amount)\b", re.I),
        _m("sum", "payments", "amount", "paid_amount"),
    ),
    MeasureRule(
        re.compile(r"\b(discounts?|markdowns?)\b", re.I),
        _m("sum", "order_items", "discount_amount", "discount_amount"),
    ),
    MeasureRule(
        re.compile(r"\b(units?\s+sold|quantity\s+sold|units?|quantities|volume|"
                   r"how\s+many\s+.{0,20}\bsold\b)\b", re.I),
        _m("sum", "order_items", "quantity", "units"),
    ),
    MeasureRule(
        re.compile(r"\b(revenue|net\s+sales|gross\s+sales|sales|gmv|turnover|spend|spent|"
                   r"how\s+much\s+.{0,25}(sold|sell|made|earn))\b", re.I),
        _m("sum", "order_items", "line_total", "revenue"),
    ),
    MeasureRule(re.compile(r"\b(list\s+price|selling\s+price|price)\b", re.I),
                _m("avg", "products", "list_price", "avg_list_price")),
    MeasureRule(re.compile(r"\b(unit\s+cost|cogs|cost\s+of\s+goods)\b", re.I),
                _m("avg", "products", "unit_cost", "avg_unit_cost")),
]

# Dimension phrases, most specific first. Alias doubles as the output column name.
DIMENSION_RULES: list[tuple[re.Pattern[str], Dimension]] = [
    (re.compile(r"\b(product\s+)?departments?\b", re.I), Dimension("categories", "department", "department")),
    (re.compile(r"\b(product\s+)?categor(y|ies)\b", re.I), Dimension("categories", "category_name", "category")),
    (re.compile(r"\b(sales\s+)?channels?\b", re.I), Dimension("orders", "channel", "channel")),
    (re.compile(r"\bfulfil?lment(\s+types?)?\b", re.I), Dimension("orders", "fulfillment_type", "fulfillment_type")),
    (re.compile(r"\b(promo(tion)?\s*codes?|coupons?)\b", re.I), Dimension("orders", "promo_code", "promo_code")),
    (re.compile(r"\bcurrenc(y|ies)\b", re.I), Dimension("orders", "currency", "currency")),
    (re.compile(r"\border\s+status\b", re.I), Dimension("orders", "status", "order_status")),
    (re.compile(r"\b(payment\s+)?methods?\b|\btender\s+types?\b", re.I), Dimension("payments", "method", "payment_method")),
    (re.compile(r"\b(processors?|gateways?|psps?)\b", re.I), Dimension("payments", "processor", "processor")),
    (re.compile(r"\bpayment\s+status\b", re.I), Dimension("payments", "status", "payment_status")),
    (re.compile(r"\b(carriers?|couriers?|shippers?)\b", re.I), Dimension("shipments", "carrier", "carrier")),
    (re.compile(r"\b(shipment|delivery)\s+status\b", re.I), Dimension("shipments", "status", "shipment_status")),
    (re.compile(r"\bdestination\s+countr(y|ies)\b", re.I), Dimension("shipments", "destination_country", "destination_country")),
    (re.compile(r"\b(return\s+)?reasons?\b|\breason\s+codes?\b", re.I), Dimension("returns", "reason", "reason")),
    (re.compile(r"\b(conditions?|dispositions?)\b", re.I), Dimension("returns", "condition", "condition")),
    (re.compile(r"\b(customer\s+)?(loyalty(\s+tiers?)?|tiers?|memberships?)\b", re.I), Dimension("customers", "loyalty_tier", "loyalty_tier")),
    (re.compile(r"\b(customer\s+)?segments?\b", re.I), Dimension("customers", "segment", "segment")),
    (re.compile(r"\bstore\s+(type|format)s?\b", re.I), Dimension("stores", "store_type", "store_type")),
    (re.compile(r"\bstates?\b|\bprovinces?\b", re.I), Dimension("stores", "state", "state")),
    (re.compile(r"\bsuppliers?\b|\bvendors?\b", re.I), Dimension("suppliers", "supplier_name", "supplier")),
    (re.compile(r"\bstores?\b|\bshops?\b|\boutlets?\b", re.I), Dimension("stores", "store_name", "store")),
    (re.compile(r"\bproducts?\b|\bskus?\b", re.I), Dimension("products", "product_name", "product")),
    (re.compile(r"\bcustomers?\b|\bbuyers?\b|\bshoppers?\b", re.I), Dimension("customers", "customer_id", "customer_id")),
]

#: Geography columns exist on several entities; the question decides which.
GEO_OWNERS = [
    (re.compile(r"\b(store|shop|outlet|branch)\b", re.I), "stores"),
    (re.compile(r"\b(supplier|vendor|sourc\w+)\b", re.I), "suppliers"),
    (re.compile(r"\b(destination|ship(ped)?\s+to)\b", re.I), "shipments"),
    (re.compile(r"\b(customer|buyer|shopper)\b", re.I), "customers"),
]
#: The owner prefix is part of the pattern so "customer region" is claimed as a
#: single phrase and the bare "customer" rule cannot fire on the same words.
_OWNER = r"(?:customer|buyer|shopper|store|shop|outlet|supplier|vendor|destination)s?\s+"
GEO_COLUMNS = [
    (re.compile(rf"\b({_OWNER})?(regions?|territor(y|ies)|geos?)\b", re.I), "region"),
    (re.compile(rf"\b({_OWNER})?countr(y|ies)\b", re.I), "country"),
    (re.compile(rf"\b({_OWNER})?(cit(y|ies)|metros?)\b", re.I), "city"),
]

#: Entity nouns that name a countable fact, used when a superlative asks
#: "the most <thing>" without naming a measure.
ENTITY_TABLES = {
    "order": "orders", "orders": "orders",
    "shipment": "shipments", "shipments": "shipments", "parcel": "shipments", "parcels": "shipments",
    "return": "returns", "returns": "returns", "refund": "returns", "refunds": "returns",
    "payment": "payments", "payments": "payments", "charge": "payments", "charges": "payments",
    "product": "products", "products": "products", "sku": "products", "skus": "products",
    "customer": "customers", "customers": "customers",
    "store": "stores", "stores": "stores",
    "supplier": "suppliers", "suppliers": "suppliers",
}
SUPERLATIVE_ENTITY = re.compile(
    r"\b(most|fewest|highest|lowest|largest|smallest|top\s+\d+|bottom\s+\d+)\s+"
    r"(?:\w+\s+){0,2}?(" + "|".join(sorted(ENTITY_TABLES, key=len, reverse=True)) + r")\b",
    re.I,
)

TIME_GRAINS: list[tuple[re.Pattern[str], tuple[str, str]]] = [
    (re.compile(r"\b(per|by|each)\s+month\b|\bmonthly\b|\bmonth\s+over\s+month\b|\bby\s+month\b", re.I), ("%Y-%m", "month")),
    (re.compile(r"\b(per|by|each)\s+quarter\b|\bquarterly\b", re.I), ("QUARTER", "quarter")),
    (re.compile(r"\b(per|by|each)\s+year\b|\byearly\b|\bannual(ly)?\b|\byear\s+over\s+year\b", re.I), ("%Y", "year")),
    (re.compile(r"\b(per|by|each)\s+day\b|\bdaily\b", re.I), ("%Y-%m-%d", "day")),
    (re.compile(r"\b(per|by|each)\s+week\b|\bweekly\b", re.I), ("%Y-%W", "week")),
    (re.compile(r"\bmonthly\s+trend\b|\btrend\s+by\s+month\b", re.I), ("%Y-%m", "month")),
]

MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06", "jul": "07",
    "aug": "08", "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12",
}

TOP_N = re.compile(r"\btop\s+(\d+)\b|\b(\d+)\s+(highest|largest|best|biggest|most)\b", re.I)
BOTTOM_N = re.compile(r"\bbottom\s+(\d+)\b|\b(\d+)\s+(lowest|smallest|worst|fewest)\b", re.I)
SINGLE_MAX = re.compile(r"\bwhich\b.{0,60}\b(most|highest|largest|biggest|best|top)\b", re.I)
SINGLE_MIN = re.compile(r"\bwhich\b.{0,60}\b(least|lowest|fewest|smallest|worst)\b", re.I)
ASC_HINT = re.compile(r"\b(lowest|fewest|smallest|worst|least|ascending)\b", re.I)

#: Date column each fact family is measured on.
DATE_SOURCES = {
    "returns": ("returns", "return_ts"),
    "shipments": ("shipments", "shipped_ts"),
    "payments": ("payments", "payment_ts"),
}
DEFAULT_DATE_SOURCE = ("orders", "order_date")


class QuestionParser:
    """Rule-based reader that produces a :class:`Plan` from a question."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._value_index = self._build_value_index()

    def _build_value_index(self) -> dict[str, list[tuple[str, str, str]]]:
        """Lowercased categorical value -> [(table, column, real value)]."""
        index: dict[str, list[tuple[str, str, str]]] = {}
        for table in self.catalog.tables.values():
            for col in table.columns:
                for value in col.sample_values:
                    for key in {value.lower(), value.lower().replace("_", " ")}:
                        index.setdefault(key, []).append((table.name, col.name, value))
        return index

    # -- refusals ----------------------------------------------------------
    def _refusal(self, q: str) -> Refusal | None:
        if WRITE_INTENT.search(q):
            return Refusal(
                "write_intent",
                "Vantage has read-only access to the warehouse and cannot modify data.",
                "Ask for the current state of the data instead, and take the change to the owning system.",
            )
        if PII_INTENT.search(q):
            return Refusal(
                "pii_request",
                "That answer would expose personally identifying customer fields (name, email, contact details).",
                "Aggregate instead: counts or totals by segment, region or loyalty tier.",
            )
        if OUT_OF_SCOPE.search(q):
            return Refusal(
                "out_of_scope",
                "The warehouse holds orders, products, customers, payments, shipments and returns only. "
                "It has no table covering what was asked.",
                f"Answerable subjects: {', '.join(self.catalog.table_names)}.",
            )
        if UNSUPPORTED.search(q):
            return Refusal(
                "unsupported_analysis",
                "Vantage reports what the warehouse recorded. It does not forecast, attribute cause or advise.",
                "Rephrase as a historical measurement, for example the same metric over the last four quarters.",
            )
        if VAGUE.search(q):
            return Refusal(
                "ambiguous",
                "The question does not name a measure or a breakdown, so any answer would be a guess.",
                "Name a metric (revenue, units, orders, refunds) and a breakdown (category, region, month).",
            )
        return None

    # -- components --------------------------------------------------------
    def _measure(self, q: str) -> tuple[Measure, tuple[int, int]] | None:
        """First matching measure rule, with the span of words it consumed.

        The span matters: in "average supplier rating by supplier country" the
        first "supplier" belongs to the measure, and only the second one is a
        breakdown.
        """
        for rule in MEASURE_RULES:
            match = rule.pattern.search(q)
            if match:
                return rule.build(), match.span()
        return None

    def _geo_dimension(self, q: str) -> tuple[Dimension, tuple[int, int]] | None:
        for pattern, column in GEO_COLUMNS:
            match = pattern.search(q)
            if not match:
                continue
            owner = "customers"
            for owner_pattern, table in GEO_OWNERS:
                if owner_pattern.search(q):
                    owner = table
                    break
            span = match.span()
            if owner == "shipments" and column == "country":
                return Dimension("shipments", "destination_country", "destination_country"), span
            if self.catalog.has_column(owner, column):
                return Dimension(owner, column, f"{owner.rstrip('s')}_{column}"), span
        return None

    def _dimensions(
        self, q: str, measure: Measure | None, claimed: list[tuple[int, int]] | None = None
    ) -> list[Dimension]:
        """Collect breakdowns, letting the longest phrase claim its words.

        Without span tracking, "revenue by product category" yields two
        dimensions: `categories.category_name` from "category" and
        `products.product_name` from "product". The compound phrase owns both
        words, so any later rule overlapping a claimed span is skipped.
        """
        found: list[Dimension] = []
        claimed = list(claimed or [])
        geo = self._geo_dimension(q)
        if geo and not any(
            geo[1][0] < c_end and geo[1][1] > c_start for c_start, c_end in claimed
        ):
            found.append(geo[0])
            claimed.append(geo[1])
        for pattern, dim in DIMENSION_RULES:
            # Take the first occurrence that no earlier phrase already claimed.
            span = next(
                (
                    m.span()
                    for m in pattern.finditer(q)
                    if not any(m.start() < c_end and m.end() > c_start for c_start, c_end in claimed)
                ),
                None,
            )
            if span is None:
                continue
            start, end = span
            if any(d.table == dim.table and d.column == dim.column for d in found):
                continue
            # A word that only named the measure is not also a breakdown:
            # "how many products" groups by nothing, "revenue by product" does.
            if measure and dim.table == measure.table and dim.column == measure.column:
                continue
            found.append(dim)
            claimed.append((start, end))
        return found[:2]

    def _date_source(self, tables: set[str]) -> tuple[str, str]:
        for table, source in DATE_SOURCES.items():
            if table in tables:
                return source
        return DEFAULT_DATE_SOURCE

    def _time_dimension(self, q: str, tables: set[str]) -> Dimension | None:
        for pattern, (fmt, alias) in TIME_GRAINS:
            if not pattern.search(q):
                continue
            table, column = self._date_source(tables)
            ref = "{%s}.%s" % (table, column)
            if fmt == "QUARTER":
                expr = (
                    f"strftime('%Y', {ref}) || '-Q' || "
                    f"CAST((CAST(strftime('%m', {ref}) AS INTEGER) + 2) / 3 AS TEXT)"
                )
            else:
                expr = f"strftime('{fmt}', {ref})"
            return Dimension(table, column, alias, expr=expr)
        return None

    def _time_filters(self, q: str, tables: set[str]) -> list[Filter]:
        table, column = self._date_source(tables)
        ref = "{%s}.%s" % (table, column)
        out: list[Filter] = []

        month_year = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", q, re.I)
        if month_year:
            month = MONTHS[month_year.group(1).lower()]
            out.append(
                Filter(table, column, "expr",
                       expr=f"strftime('%Y-%m', {ref}) = '{month_year.group(2)}-{month}'",
                       description=f"{month_year.group(1).title()} {month_year.group(2)} only")
            )
            return out

        years = sorted({y for y in re.findall(r"\b(20\d\d)\b", q)})
        if len(years) == 1:
            out.append(
                Filter(table, column, "expr", expr=f"strftime('%Y', {ref}) = '{years[0]}'",
                       description=f"calendar year {years[0]}")
            )
        elif len(years) > 1:
            joined = ", ".join(f"'{y}'" for y in years)
            out.append(
                Filter(table, column, "expr", expr=f"strftime('%Y', {ref}) IN ({joined})",
                       description=f"calendar years {', '.join(years)}")
            )
        return out

    def _status_filters(self, q: str, tables: set[str]) -> list[Filter]:
        out: list[Filter] = []
        if re.search(r"\bcompleted\b", q, re.I):
            out.append(Filter("orders", "status", "=", "completed", description="completed orders only"))
        elif re.search(r"\bcancell?ed\b", q, re.I):
            out.append(Filter("orders", "status", "=", "cancelled", description="cancelled orders only"))
        if re.search(r"\bdelivered\b", q, re.I) and "shipments" in tables:
            out.append(Filter("shipments", "status", "=", "delivered", description="delivered shipments only"))
        if re.search(r"\blost\b", q, re.I) and "shipments" in tables:
            out.append(Filter("shipments", "status", "=", "lost", description="lost shipments only"))
        if re.search(r"\b(failed|declined)\b", q, re.I) and "payments" in tables:
            out.append(Filter("payments", "status", "=", "failed", description="failed payments only"))
        if re.search(r"\binactive\b|\bdiscontinued\b", q, re.I):
            table = "products" if re.search(r"\bproducts?\b|\bskus?\b", q, re.I) else "customers"
            out.append(Filter(table, "is_active", "=", 0, description=f"inactive {table} only"))
        elif re.search(r"\bactive\b", q, re.I):
            table = "products" if re.search(r"\bproducts?\b|\bskus?\b", q, re.I) else "customers"
            out.append(Filter(table, "is_active", "=", 1, description=f"active {table} only"))
        if re.search(r"\bperishable\b", q, re.I):
            out.append(Filter("categories", "is_perishable", "=", 1, description="perishable categories only"))
        return out

    def _value_filters(self, q: str, in_play: set[str], dimensions: list[Dimension]) -> list[Filter]:
        """Literal category values mentioned in the question ('Platinum', 'FedEx').

        Only applied when the owning table is already in the query, or when the
        question names that entity: 'quality' is a return reason *and* an English
        word, and a supplier-quality question must not acquire a returns filter.
        """
        lowered = q.lower()
        taken = {(d.table, d.column) for d in dimensions}
        out: list[Filter] = []
        for key, owners in self._value_index.items():
            if len(key) < 3 or not re.search(rf"\b{re.escape(key)}\b", lowered):
                continue
            for table, column, value in owners:
                if (table, column) in taken or table not in in_play:
                    continue
                if column in ("region", "country", "city") and table not in in_play:
                    continue
                if any(f.table == table and f.column == column for f in out):
                    continue
                out.append(Filter(table, column, "=", value, description=f"{column} = {value}"))
        return out

    def _geo_value_filters(self, q: str, dimensions: list[Dimension]) -> list[Filter]:
        """'revenue in Europe' -> a region filter on the entity the question implies."""
        lowered = q.lower()
        owner = "customers"
        for owner_pattern, table in GEO_OWNERS:
            if owner_pattern.search(q):
                owner = table
                break
        taken = {(d.table, d.column) for d in dimensions}
        out: list[Filter] = []
        for column in ("region", "country", "city"):
            if not self.catalog.has_column(owner, column) or (owner, column) in taken:
                continue
            col = self.catalog.table(owner).column(column)
            for value in col.sample_values:
                if re.search(rf"\b{re.escape(value.lower())}\b", lowered):
                    out.append(Filter(owner, column, "=", value, description=f"{owner}.{column} = {value}"))
                    break
        return out[:1]

    def _superlative_count(self, q: str, dimensions: list[Dimension]) -> Measure | None:
        """"Which carrier has the most lost shipments?" names no measure but is
        plainly a count of the entity the superlative points at."""
        if not (SINGLE_MAX.search(q) or SINGLE_MIN.search(q) or TOP_N.search(q) or BOTTOM_N.search(q)):
            return None
        match = SUPERLATIVE_ENTITY.search(q)
        table = ENTITY_TABLES.get(match.group(2).lower()) if match else None
        if table is None and dimensions:
            table = dimensions[0].table
        if table is None or not self.catalog.has_table(table):
            return None
        pk = self.catalog.table(table).primary_key
        column = pk[0] if pk else "*"
        agg = "count_distinct" if any(d.table != table for d in dimensions) else "count"
        return Measure(agg=agg, table=table, column=column, alias=f"{table.rstrip('s')}_count")

    def _ordering(self, q: str, measure: Measure | None) -> tuple[str, int | None]:
        top = TOP_N.search(q)
        bottom = BOTTOM_N.search(q)
        if bottom:
            n = next(g for g in bottom.groups() if g and g.isdigit())
            return "measure_asc", int(n)
        if top:
            n = next(g for g in top.groups() if g and g.isdigit())
            return "measure_desc", int(n)
        if SINGLE_MIN.search(q):
            return "measure_asc", 1
        if SINGLE_MAX.search(q):
            return "measure_desc", 1
        if not measure:
            return "dimension_asc", None
        return ("measure_asc" if ASC_HINT.search(q) else "measure_desc"), None

    # -- entry point -------------------------------------------------------
    def parse(self, question: str) -> Plan:
        q = question.strip()
        refusal = self._refusal(q)
        if refusal:
            return Plan(intent="refuse", refusal=refusal, rationale=refusal.reason)

        measured = self._measure(q)
        measure = measured[0] if measured else None
        claimed = [measured[1]] if measured else []
        dimensions = self._dimensions(q, measure, claimed)
        if measure is None:
            measure = self._superlative_count(q, dimensions)
            dimensions = [d for d in dimensions if not (measure and d.table == measure.table
                                                        and d.column == measure.column)]

        if measure is None and not dimensions:
            return Plan(
                intent="refuse",
                refusal=Refusal(
                    "ambiguous",
                    "No measure or breakdown could be identified in the question.",
                    "Name a metric (revenue, units, orders, refunds) and optionally a breakdown.",
                ),
                rationale="unparseable question",
            )

        in_play = set()
        if measure:
            in_play.add(measure.table)
            in_play.update(measure.extra_tables)
        in_play.update(d.table for d in dimensions)

        time_dim = self._time_dimension(q, in_play)
        if time_dim:
            dimensions = [time_dim, *dimensions][:3]
            in_play.add(time_dim.table)

        filters: list[Filter] = []
        filters += self._time_filters(q, in_play)
        filters += self._status_filters(q, in_play)
        filters += self._geo_value_filters(q, dimensions)
        filters += self._value_filters(q, in_play | {f.table for f in filters}, dimensions)

        filters = self._dedupe(filters)
        order_by, limit = self._ordering(q, measure)
        notes: list[str] = []
        optional: list[str] = []
        if measure and measure.agg == "ratio" and "returns" in measure.extra_tables:
            optional.append("returns")
            notes.append("returns is LEFT JOINed so unreturned lines stay in the denominator")
        if measure and measure.alias == "avg_delivery_days":
            filters.append(
                Filter("shipments", "delivered_ts", "is not null",
                       description="delivered shipments only; undelivered parcels have no duration")
            )

        return Plan(
            intent="aggregate" if measure else "list",
            measure=measure,
            dimensions=dimensions,
            filters=filters,
            order_by=order_by,
            limit=limit,
            optional_tables=optional,
            notes=notes,
            rationale=self._rationale(measure, dimensions, filters),
        )

    @staticmethod
    def _dedupe(filters: list[Filter]) -> list[Filter]:
        """Keep the first predicate per (table, column, expr); later passes repeat."""
        seen: set[tuple] = set()
        out: list[Filter] = []
        for flt in filters:
            key = (flt.table, flt.column, flt.expr or flt.op)
            if key in seen:
                continue
            seen.add(key)
            out.append(flt)
        return out

    @staticmethod
    def _rationale(measure: Measure | None, dimensions: list[Dimension], filters: list[Filter]) -> str:
        head = f"{measure.agg}({measure.table}.{measure.column})" if measure else "row listing"
        by = f" by {', '.join(d.alias for d in dimensions)}" if dimensions else ""
        where = f" filtered on {', '.join(f.description or f.column for f in filters)}" if filters else ""
        return f"{head}{by}{where}".strip()


# --------------------------------------------------------------------------
# Fault injection for the self-correction tier.
# --------------------------------------------------------------------------
FAULTS: dict[str, Callable[[str], str]] = {
    "unknown_column": lambda sql: re.sub(r"\bline_total\b", "total_amount", sql, count=1),
    "unknown_table": lambda sql: re.sub(r"\border_items\b", "sales_facts", sql),
    "missing_group_by": lambda sql: re.sub(r"\nGROUP BY [^\n]+", "", sql, count=1),
    "bad_join_column": lambda sql: re.sub(r"\.product_id\b", ".prod_id", sql, count=1),
    "write_statement": lambda sql: "DELETE FROM order_items WHERE 1=1",
    "syntax_error": lambda sql: sql.replace("SELECT", "SELEKT", 1),
    "stacked_statement": lambda sql: sql + "; DROP TABLE orders",
}


class MockAnalyst:
    """Deterministic :class:`~vantage.llm.base.LLMClient` used as the bench control."""

    def __init__(
        self,
        catalog: Catalog,
        fault_profile: str | None = None,
        unfaithful_memo: bool = False,
    ) -> None:
        if fault_profile and fault_profile not in FAULTS:
            raise ValueError(f"unknown fault profile {fault_profile!r}; known: {sorted(FAULTS)}")
        self.catalog = catalog
        self.parser = QuestionParser(catalog)
        self.compiler = SqlCompiler(catalog)
        self.fault_profile = fault_profile
        self.unfaithful_memo = unfaithful_memo
        self.name = "mock" + (f"+fault:{fault_profile}" if fault_profile else "")

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        handler = {
            "plan": self._plan,
            "sql": self._sql,
            "critique": self._critique,
            "memo": self._memo,
        }[request.task]
        text = handler(request.payload)
        return LLMResponse(text=text, model=self.name, latency_ms=(time.perf_counter() - started) * 1000)

    # -- node handlers -----------------------------------------------------
    def _plan(self, payload: dict) -> str:
        plan = self.parser.parse(payload.get("question", ""))
        return json.dumps(plan.to_dict(), indent=2)

    def _sql(self, payload: dict) -> str:
        plan = Plan.from_dict(payload.get("plan") or {})
        try:
            sql = self.compiler.compile(plan).sql
        except CompileError as err:
            return f"-- compile failed: {err}\nSELECT 'unanswerable' AS error"

        attempt = int(payload.get("attempt", 1))
        critique = payload.get("critique") or {}
        # The injected model only fails the first time, and only repairs once the
        # critic has actually diagnosed something. A blind retry stays broken.
        if self.fault_profile and attempt == 1 and not critique.get("repair_hint"):
            return FAULTS[self.fault_profile](sql)
        return sql

    def _critique(self, payload: dict) -> str:
        """Structured verdict. The rule-based critic in the graph does the real work;
        this only mirrors it so a hosted model and the baseline share one contract."""
        return json.dumps(
            {
                "verdict": payload.get("suggested_verdict", "accept"),
                "reason": payload.get("suggested_reason", ""),
                "repair_hint": payload.get("suggested_hint", ""),
            }
        )

    def _memo(self, payload: dict) -> str:
        question = payload.get("question", "")
        columns: list[str] = payload.get("columns", []) or []
        rows: list[list[Any]] = payload.get("rows", []) or []
        row_count = int(payload.get("row_count", len(rows)))

        if not rows:
            return json.dumps(
                {
                    "headline": "The query ran successfully and returned no rows.",
                    "claims": [],
                    "caveats": ["No rows matched the filters, so there is nothing to summarise."],
                }
            )

        value_idx = len(columns) - 1
        label_idx = 0 if len(columns) > 1 else None
        claims = []
        for i, row in enumerate(rows[:3]):
            value = row[value_idx]
            label = str(row[label_idx]) if label_idx is not None else "overall"
            claims.append(
                {
                    "text": f"{label} recorded {value} for {columns[value_idx].replace('_', ' ')}.",
                    "value": value,
                    "row": i,
                    "column": columns[value_idx],
                }
            )
        if self.unfaithful_memo:
            # Control case for the faithfulness tier: a plausible number that is
            # not in the result set at all.
            claims.append(
                {
                    "text": "Across the period this represents a 42.7% share of the total.",
                    "value": 42.7,
                    "row": 0,
                    "column": columns[value_idx],
                }
            )

        lead_label = str(rows[0][label_idx]) if label_idx is not None else "the result"
        metric = columns[value_idx].replace("_", " ")
        headline = (
            f"{row_count} row(s) returned; {lead_label} leads on {metric} with {rows[0][value_idx]}."
            if label_idx is not None
            else f"{metric} is {rows[0][value_idx]}."
        )
        return json.dumps(
            {
                "headline": headline,
                "claims": claims,
                "caveats": payload.get("caveats", []) or [],
            },
            default=str,
        )
