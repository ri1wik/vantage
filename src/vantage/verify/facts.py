"""Verified-facts checker: every number in a memo must come from the result set.

This is the last hallucination control in the pipeline, and the one that catches
the failure the others cannot. The guard proves the SQL is safe and the executor
proves it ran; neither proves the prose *about* the answer is true. A model that
executes a correct query and then writes "roughly a third of revenue" from
nowhere has produced a confident, well-sourced, wrong memo.

So the checker re-derives what a memo is allowed to say. A number is grounded
when it appears in a result cell, or is one of the aggregates the checker
computes itself from that result: the row count, a column total, or one row's
share of a column total. Anything else is unverified and gets stripped before
the memo is shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from ..executor import QueryResult

#: Numbers written the way memos write them: 1,234.50 / 42% / $1.2 / -3.
#: The trailing lookahead matters: without it the grouped branch matches the
#: first three digits of an ungrouped number like 1143439.48 and the rest of the
#: figure is silently dropped from the check.
NUMBER = re.compile(
    r"(?<![\w.])[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\d,])"
)

#: Absolute slack for a rounded restatement of a grounded value.
ABS_TOLERANCE = 0.005


@dataclass(frozen=True)
class GroundedValue:
    value: float
    source: str          # cell | row_count | column_total | share_pct | query_literal
    column: str = ""
    row: int | None = None

    def describe(self) -> str:
        if self.source == "cell":
            return f"{self.column}[row {self.row}]"
        if self.source == "column_total":
            return f"SUM({self.column})"
        if self.source == "share_pct":
            return f"{self.column}[row {self.row}] as % of SUM({self.column})"
        if self.source == "query_literal":
            return f"literal {self.value:g} in the executed SQL"
        return self.source


@dataclass
class Claim:
    """One numeric assertion lifted out of a memo."""

    literal: str
    value: float
    verified: bool = False
    source: str = ""

    def as_dict(self) -> dict:
        return {"literal": self.literal, "value": self.value, "verified": self.verified, "source": self.source}


@dataclass
class FactCheck:
    claims: list[Claim] = field(default_factory=list)
    grounded_count: int = 0

    @property
    def unverified(self) -> list[Claim]:
        return [c for c in self.claims if not c.verified]

    @property
    def faithfulness(self) -> float:
        """Share of numeric claims traceable to the result set. 1.0 when there are none."""
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.verified) / len(self.claims)

    @property
    def ok(self) -> bool:
        return not self.unverified

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "checked": len(self.claims),
            "unverified": [c.literal for c in self.unverified],
            "claims": [c.as_dict() for c in self.claims],
        }


def _to_float(literal: str) -> float | None:
    text = literal.strip().replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _decimals(literal: str) -> int:
    body = literal.rstrip("%")
    return len(body.split(".")[1]) if "." in body else 0


def ground(result: QueryResult) -> list[GroundedValue]:
    """Every number a memo about ``result`` is permitted to state."""
    grounded: list[GroundedValue] = [GroundedValue(float(result.row_count), "row_count")]

    numeric_columns: dict[str, list[tuple[int, float]]] = {}
    for r, row in enumerate(result.rows):
        for c, cell in enumerate(row):
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                continue
            column = result.columns[c] if c < len(result.columns) else f"col{c}"
            grounded.append(GroundedValue(float(cell), "cell", column, r))
            numeric_columns.setdefault(column, []).append((r, float(cell)))

    for column, entries in numeric_columns.items():
        total = sum(v for _, v in entries)
        grounded.append(GroundedValue(total, "column_total", column))
        if total:
            for r, value in entries:
                grounded.append(GroundedValue(value / total * 100.0, "share_pct", column, r))
    return grounded


def ground_sql(sql: str) -> list[GroundedValue]:
    """Numeric literals from the executed query.

    A memo is allowed to restate the scope it was given: "revenue in 2024" cites
    a number that came from the question and is now a literal in a query the
    reader can see. Those are grounded by construction. Numbers that appear in
    neither the result nor the query are the ones worth catching.
    """
    if not sql.strip():
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # pragma: no cover - the guard has already parsed this
        return []
    out: list[GroundedValue] = []
    for node in tree.find_all(exp.Literal):
        text = str(node.name)
        for token in re.findall(r"\d+(?:\.\d+)?", text):
            try:
                out.append(GroundedValue(float(token), "query_literal"))
            except ValueError:  # pragma: no cover
                continue
    return out


def _labels(result: QueryResult) -> list[str]:
    """String cells, longest first, so '2025-03' is masked before '2025'."""
    seen = {str(cell) for row in result.rows for cell in row if isinstance(cell, str) and cell.strip()}
    return sorted(seen, key=len, reverse=True)


def _mask_labels(text: str, result: QueryResult) -> str:
    """Blank out label text so a dimension value like '2024-06' is not read as a claim."""
    for label in _labels(result):
        text = text.replace(label, " ")
    return text


def check_text(text: str, result: QueryResult, sql: str = "") -> FactCheck:
    """Verify every number in ``text`` against ``result`` and the query that made it."""
    grounded = ground(result) + ground_sql(sql or result.sql)
    masked = _mask_labels(text, result)
    check = FactCheck(grounded_count=len(grounded))

    for match in NUMBER.finditer(masked):
        literal = match.group(0)
        value = _to_float(literal)
        if value is None:
            continue
        precision = _decimals(literal)
        hit = next(
            (
                g
                for g in grounded
                if abs(round(g.value, precision) - value) <= ABS_TOLERANCE
                or abs(g.value - value) <= ABS_TOLERANCE
            ),
            None,
        )
        check.claims.append(
            Claim(literal=literal, value=value, verified=hit is not None, source=hit.describe() if hit else "")
        )
    return check


def check_memo(memo: dict[str, Any], result: QueryResult, sql: str = "") -> FactCheck:
    """Verify a structured memo: the prose *and* each claim's stated provenance.

    A claim carrying ``row``/``column`` is held to that exact cell, not merely to
    "some cell somewhere", so a memo cannot cite the right number against the
    wrong row.
    """
    prose = " ".join(
        [str(memo.get("headline", ""))]
        + [str(c.get("text", "")) for c in memo.get("claims", []) or []]
        + [str(c) for c in memo.get("caveats", []) or []]
    )
    check = check_text(prose, result, sql=sql)

    for claim in memo.get("claims", []) or []:
        row, column = claim.get("row"), claim.get("column")
        stated = claim.get("value")
        if row is None or column is None or stated is None:
            continue
        literal = f"claim@{column}[row {row}]"
        actual = _cell(result, int(row), str(column))
        verified = actual is not None and _close(actual, stated)
        check.claims.append(
            Claim(
                literal=literal,
                value=float(stated) if isinstance(stated, (int, float)) else 0.0,
                verified=verified,
                source=f"{column}[row {row}]" if verified else "",
            )
        )
    return check


def _cell(result: QueryResult, row: int, column: str) -> Any:
    if not (0 <= row < len(result.rows)) or column not in result.columns:
        return None
    return result.rows[row][result.columns.index(column)]


def _close(actual: Any, stated: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(stated, (int, float)):
        return abs(float(actual) - float(stated)) <= ABS_TOLERANCE
    return str(actual) == str(stated)


def redact(text: str, check: FactCheck, marker: str = "[unverified]") -> str:
    """Replace every unverified literal so an unfaithful memo cannot ship a number."""
    out = text
    for claim in check.unverified:
        if claim.literal.startswith("claim@"):
            continue
        out = out.replace(claim.literal, marker)
    return out
