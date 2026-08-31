"""The verified-facts checker: the control on what a memo is allowed to say."""

from __future__ import annotations

from vantage.executor import QueryResult
from vantage.verify.facts import NUMBER, check_memo, check_text, redact

RESULT = QueryResult(
    columns=["category", "revenue"],
    rows=[("Storage", 1183198.26), ("Cycling", 1143439.48), ("Figures", 1101915.01)],
    elapsed_ms=1.0,
    sql="SELECT category, revenue FROM t WHERE year = '2024'",
)


def test_ungrouped_numbers_are_tokenised_whole() -> None:
    """The grouped branch must not swallow the first three digits of 1143439.48."""
    found = [m.group(0) for m in NUMBER.finditer("1,183,198.26 and 1143439.48 over 3 rows at 42.7%")]
    assert found == ["1,183,198.26", "1143439.48", "3", "42.7%"]


def test_cell_values_row_counts_and_totals_are_grounded() -> None:
    text = "Across 3 rows, Storage led with 1,183,198.26 of the 3,428,552.75 total."
    assert check_text(text, RESULT).faithfulness == 1.0


def test_a_derived_share_is_grounded_but_an_invented_one_is_not() -> None:
    assert check_text("Storage was 34.51% of the total.", RESULT).ok
    check = check_text("Storage was roughly 42.7% of the total.", RESULT)
    assert not check.ok
    assert [c.literal for c in check.unverified] == ["42.7%"]


def test_query_literals_count_as_grounded() -> None:
    """A memo may restate the scope it was given; that number is in the SQL."""
    assert check_text("Revenue in 2024 was led by Storage.", RESULT).ok


def test_a_claim_is_held_to_its_stated_row_and_column() -> None:
    memo = {
        "headline": "Storage leads.",
        "claims": [{"text": "Cycling recorded 1143439.48.", "value": 1143439.48, "row": 2, "column": "revenue"}],
    }
    check = check_memo(memo, RESULT, sql=RESULT.sql)
    assert "claim@revenue[row 2]" in [c.literal for c in check.unverified]
    memo["claims"][0]["row"] = 1
    assert check_memo(memo, RESULT, sql=RESULT.sql).ok


def test_unverified_figures_are_redacted_from_the_prose() -> None:
    text = "Storage led with 1,183,198.26, about 42.7% of revenue."
    redacted = redact(text, check_text(text, RESULT))
    assert "42.7%" not in redacted
    assert "1,183,198.26" in redacted
