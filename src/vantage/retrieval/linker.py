"""Hybrid schema linker: TF-IDF retrieval + FK-closure expansion.

Feeding the whole schema to the SQL writer is the cheapest way to get
hallucinated joins, and feeding it too little is the cheapest way to get an
unanswerable question. The linker splits the difference:

1. **Lexicon pass** pins the fact table implied by the measure ("revenue",
   "refund", "delivery") before any statistical scoring runs.
2. **TF-IDF pass** scores every table document (name, description, grain,
   synonyms, column docs, low-cardinality sample values) against the question.
3. **FK-closure pass** adds only the bridge tables that sit on the shortest join
   path between two selected tables, so joins are always spellable.
4. **Column-ownership pass** catches the case statistical scoring is worst at:
   a question naming one distinctive column ("revenue by currency") whose owning
   table scores far below the fact table. If a named column exists on no selected
   table, the table that owns it is added.
5. **Temporal anchor pass** handles the one structural trap in this warehouse:
   ``order_items`` is the revenue fact but carries no date, so any time-grained
   revenue question also needs ``orders``. Rather than hardcode that pair, the
   linker adds the nearest neighbour that owns a date column whenever the
   question asks for a time grain and a selected table has none.

The bench reports linker recall against gold table sets; the design target is
100% recall with the smallest table set that still achieves it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..warehouse.catalog import Catalog
from .tfidf import TfidfIndex, normalize

# Measure and entity cues that pin a fact table regardless of TF-IDF ranking.
# Phrases are matched against the normalized question.
LEXICON: dict[str, tuple[str, float]] = {
    "revenue": ("order_items", 1.0),
    "sale": ("order_items", 0.9),
    "gmv": ("order_items", 1.0),
    "turnover": ("order_items", 0.9),
    "net revenue": ("order_items", 1.0),
    "line total": ("order_items", 1.0),
    "unit sold": ("order_items", 0.9),
    "sold": ("order_items", 0.9),
    "selling": ("order_items", 0.8),
    "unit": ("order_items", 0.5),
    "bought": ("order_items", 0.8),
    "spend": ("order_items", 0.8),
    "spent": ("order_items", 0.8),
    "quantity": ("order_items", 0.7),
    "discount": ("order_items", 0.8),
    "aov": ("order_items", 0.8),
    "average order value": ("order_items", 0.9),
    "basket size": ("order_items", 0.8),
    "margin": ("order_items", 0.7),
    "refund": ("returns", 1.0),
    "return": ("returns", 0.9),
    "returned": ("returns", 1.0),
    "rma": ("returns", 1.0),
    "return rate": ("returns", 1.0),
    "shipment": ("shipments", 1.0),
    "shipping": ("shipments", 1.0),
    "shipped": ("shipments", 0.9),
    "delivery": ("shipments", 1.0),
    "delivered": ("shipments", 1.0),
    "carrier": ("shipments", 1.0),
    "courier": ("shipments", 1.0),
    "parcel": ("shipments", 0.9),
    "freight": ("shipments", 0.9),
    "payment": ("payments", 1.0),
    "paid": ("payments", 0.8),
    "gateway": ("payments", 1.0),
    "processor": ("payments", 0.9),
    "declined": ("payments", 0.9),
    "capture": ("payments", 0.8),
    "tender": ("payments", 0.9),
    "order": ("orders", 0.7),
    "checkout": ("orders", 0.7),
    "channel": ("orders", 0.8),
    "promo": ("orders", 0.8),
    "coupon": ("orders", 0.8),
    "customer": ("customers", 0.8),
    "buyer": ("customers", 0.8),
    "shopper": ("customers", 0.8),
    "loyalty": ("customers", 0.9),
    "segment": ("customers", 0.7),
    "signup": ("customers", 0.9),
    "product": ("products", 0.8),
    "sku": ("products", 0.9),
    "item": ("products", 0.5),
    "catalog": ("products", 0.8),
    "list price": ("products", 0.9),
    "cogs": ("products", 0.9),
    "unit cost": ("products", 0.9),
    "category": ("categories", 0.9),
    "department": ("categories", 0.9),
    "supplier": ("suppliers", 0.9),
    "vendor": ("suppliers", 0.9),
    "lead time": ("suppliers", 0.9),
    "store": ("stores", 0.8),
    "shop": ("stores", 0.7),
    "outlet": ("stores", 0.7),
    "flagship": ("stores", 0.8),
}

# Lexicon hits at or above this weight are pinned as seeds; weaker hits only
# add score, so a passing mention of "item" cannot force a table into the set.
SEED_THRESHOLD = 0.7

# Region/city/country are shared column names. Whichever entity the question
# names wins; without a hint the customer dimension is the safer default.
AMBIGUOUS_GEO = ("region", "country", "city")

# Time-grain cues. A hit means the query needs a date column somewhere in scope.
TEMPORAL_CUES = re.compile(
    r"\b(month|monthly|quarter|quarterly|year|yearly|annual|annually|week|weekly|"
    r"daily|day|date|trend|over time|ytd|mtd|yoy|mom|since|20\d\d|"
    r"last (?:year|month|quarter|week)|this (?:year|month|quarter))\b"
)

# Column names that can carry a time grain.
DATE_HINTS = ("date", "_ts", "_at", "time")


def _has_date_column(table) -> bool:
    return any(any(h in c.name for h in DATE_HINTS) for c in table.columns)


@dataclass
class LinkedSchema:
    """The subset of the warehouse handed to the SQL writer."""

    tables: list[str]
    seeds: list[str] = field(default_factory=list)
    bridges: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    lexicon_hits: list[str] = field(default_factory=list)
    ddl: str = ""

    def as_dict(self) -> dict:
        return {
            "tables": self.tables,
            "seeds": self.seeds,
            "bridges": self.bridges,
            "lexicon_hits": self.lexicon_hits,
            "scores": {k: round(v, 4) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
        }


class SchemaLinker:
    """Selects the smallest join-connected table set that can answer a question."""

    def __init__(self, catalog: Catalog, top_k: int = 4, score_floor: float = 0.045) -> None:
        self.catalog = catalog
        self.top_k = top_k
        self.score_floor = score_floor
        names = list(catalog.table_names)
        self.index = TfidfIndex(labels=names, documents=[catalog.tables[n].doc() for n in names])

    # -- passes ------------------------------------------------------------
    def _lexicon(self, question_norm: str) -> dict[str, float]:
        hits: dict[str, float] = {}
        for phrase, (table, weight) in LEXICON.items():
            if normalize(phrase) in question_norm:
                hits[table] = max(hits.get(table, 0.0), weight)
        # "region"/"country"/"city" alone should not drag in stores and suppliers.
        if any(g in question_norm for g in AMBIGUOUS_GEO) and not hits:
            hits["customers"] = 0.6
        return hits

    def _lexical_boost(self, question_norm: str) -> dict[str, float]:
        """Reward literal table-name and column-name mentions."""
        boost: dict[str, float] = {}
        q_tokens = set(question_norm.split())
        for name, table in self.catalog.tables.items():
            score = 0.0
            if normalize(name) in question_norm:
                score += 0.5
            for syn in table.synonyms:
                if normalize(syn) in question_norm:
                    score += 0.2
            for col in table.columns:
                if col.name.endswith("_id"):
                    continue
                if normalize(col.name) in question_norm:
                    score += 0.15
                if q_tokens & {normalize(s) for s in col.synonyms}:
                    score += 0.1
            if score:
                boost[name] = min(score, 1.2)
        return boost

    def _column_owners(self, question_norm: str, selected: set[str]) -> set[str]:
        """Add the table owning a column the question names but no seed has.

        TF-IDF ranks whole tables, so a ten-word question dominated by one strong
        cue ("revenue") can bury the table holding the column it asks to break
        down by ("currency"). This pass is exact rather than statistical: the
        column is named, it exists in one place, and that place has to be in scope.
        """
        # Collect every column phrase the question contains, longest first: a
        # question saying "refund amount" has named `returns.refund_amount`, not
        # `payments.amount`, and the longer phrase must claim those words.
        candidates: list[tuple[int, int, str]] = []
        for table in self.catalog.tables.values():
            for col in table.columns:
                if col.name.endswith("_id") or len(col.name) < 4:
                    continue
                for phrase in [col.name, *col.synonyms]:
                    start = question_norm.find(normalize(phrase))
                    if start >= 0:
                        candidates.append((start, start + len(normalize(phrase)), col.name))
        candidates.sort(key=lambda c: (c[0] - c[1], c[0]))

        added: set[str] = set()
        claimed: list[tuple[int, int]] = []
        for start, end, column in candidates:
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            owners = self.catalog.resolve_column(column)
            if any(o in selected | added for o in owners):
                continue
            if len(owners) == 1:
                added.add(owners[0])
            elif owners:
                    # Ambiguous column name: take the owner closest to a seed, so
                    # "status" resolves against the tables already in play.
                ranked = sorted(
                    owners,
                    key=lambda o: (
                        min((len(self.catalog.join_path(seed, o)) or 99) for seed in selected)
                        if selected
                        else 99,
                        o,
                    ),
                )
                added.add(ranked[0])
        return added - selected

    def _temporal_anchors(self, question_norm: str, selected: set[str]) -> set[str]:
        """Pull in a date-bearing neighbour when a time grain is asked of a table
        that has no date of its own (the classic ``order_items`` / ``orders`` split)."""
        if not TEMPORAL_CUES.search(question_norm):
            return set()
        anchors: set[str] = set()
        for name in selected:
            table = self.catalog.table(name)
            if table is None or _has_date_column(table):
                continue
            candidates = [
                n
                for n in sorted(self.catalog.neighbours(name))
                if (t := self.catalog.table(n)) is not None and _has_date_column(t)
            ]
            if not candidates:
                continue
            # A fact's time axis lives on its event header, not on a dimension:
            # prefer a table this one points at, and among those the one closest
            # in grain. `launch_date` on a 2.5K-row product dimension is an
            # attribute; `order_date` on a 42K-row header is the axis.
            parents = {fk.to_table for fk in table.foreign_keys}
            candidates.sort(key=lambda n: (n not in parents, -self.catalog.tables[n].row_count, n))
            # Only the best anchor counts as coverage. `products` being in scope
            # does not give a revenue question a usable time axis.
            if candidates[0] not in selected:
                anchors.add(candidates[0])
        return anchors - selected

    def _bridges(self, selected: set[str]) -> set[str]:
        """Tables required to spell the joins between the selected tables."""
        bridges: set[str] = set()
        ordered = sorted(selected)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                for fk in self.catalog.join_path(a, b):
                    bridges.update({fk.from_table, fk.to_table})
        return bridges - selected

    # -- entry point -------------------------------------------------------
    def link(self, question: str) -> LinkedSchema:
        q = normalize(question)
        tfidf = self.index.score(q)
        lexicon = self._lexicon(q)
        lexical = self._lexical_boost(q)

        combined = {
            name: tfidf.get(name, 0.0) + 1.5 * lexicon.get(name, 0.0) + lexical.get(name, 0.0)
            for name in self.catalog.table_names
        }

        ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
        best = ranked[0][1] if ranked else 0.0
        seeds = {name for name, score in ranked[: self.top_k] if score >= max(self.score_floor, best * 0.35)}
        seeds |= {t for t, w in lexicon.items() if w >= SEED_THRESHOLD}
        if not seeds and ranked:
            seeds = {ranked[0][0]}

        owners = self._column_owners(q, seeds)
        anchors = self._temporal_anchors(q, seeds | owners)
        extra = owners | anchors
        bridges = self._bridges(seeds | extra) | extra
        tables = sorted(seeds | bridges)

        return LinkedSchema(
            tables=tables,
            seeds=sorted(seeds),
            bridges=sorted(bridges),
            scores=combined,
            lexicon_hits=sorted(lexicon),
            ddl=self.catalog.render(tables),
        )

    def recall(self, question: str, gold_tables: list[str]) -> float:
        """Fraction of ``gold_tables`` present in the linked set. 1.0 is the target."""
        if not gold_tables:
            return 1.0
        linked = set(self.link(question).tables)
        return len([t for t in gold_tables if t in linked]) / len(gold_tables)
