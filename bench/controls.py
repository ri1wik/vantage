"""Null-agent and oracle controls.

A benchmark that only ever runs the system it was written for cannot tell you
whether it measures the system or the harness. These two controls bracket it:

* **null** always emits a trivially valid query and a memo full of invented
  figures. It must score near zero. If it does not, a tier is passing something
  for free, and that tier is measuring nothing.
* **oracle** executes the hand-written gold SQL through the real graph. It bounds
  what any model could score on the answer-correctness tiers, and separates "the
  model was wrong" from "the case was unanswerable as written".

The oracle deliberately scores badly on the refusal tier: it has no gold SQL for
a question that should be refused, so it answers where it should decline. That is
the expected shape, not a defect.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from vantage.executor import QueryResult
from vantage.llm.base import LLMRequest, LLMResponse
from vantage.plan import Measure, Plan
from vantage.warehouse.catalog import Catalog

CASES_PATH = Path(__file__).parent / "cases.yaml"


class NullClient:
    """The lower bound: valid SQL, no understanding, fabricated prose."""

    name = "null"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        text = {
            "plan": self._plan,
            "sql": self._sql,
            "critique": self._critique,
            "memo": self._memo,
        }[request.task](request.payload)
        return LLMResponse(text=text, model=self.name, latency_ms=(time.perf_counter() - started) * 1000)

    def _plan(self, payload: dict) -> str:
        # Always the same shape, never a refusal: a model that has stopped reading.
        plan = Plan(
            intent="aggregate",
            measure=Measure("count", "orders", "order_id", "answer"),
            rationale="null control: counts orders regardless of the question",
        )
        return json.dumps(plan.to_dict())

    def _sql(self, payload: dict) -> str:
        return "SELECT COUNT(*) AS answer FROM orders LIMIT 1"

    def _critique(self, payload: dict) -> str:
        return json.dumps({"verdict": "accept", "reason": "null control accepts everything"})

    def _memo(self, payload: dict) -> str:
        return json.dumps(
            {
                "headline": "Revenue grew 18.4% year on year, driven by a 7,250,000.00 uplift in the top category.",
                "claims": [
                    {"text": "The leading group contributed 61.3% of the total.", "value": 61.3, "row": 0,
                     "column": (payload.get("columns") or ["answer"])[-1]}
                ],
                "caveats": [],
            }
        )


class OracleClient:
    """The upper bound: the hand-written gold query, run through the real graph."""

    name = "oracle"

    def __init__(self, catalog: Catalog, cases_path: Path = CASES_PATH) -> None:
        self.catalog = catalog
        cases = yaml.safe_load(cases_path.read_text()) or []
        self.gold: dict[str, str] = {
            c["question"].strip().lower(): c["gold_sql"] for c in cases if c.get("gold_sql")
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        text = {
            "plan": self._plan,
            "sql": self._sql,
            "critique": lambda p: json.dumps({"verdict": "accept", "reason": "oracle"}),
            "memo": self._memo,
        }[request.task](request.payload)
        return LLMResponse(text=text, model=self.name, latency_ms=(time.perf_counter() - started) * 1000)

    def _plan(self, payload: dict) -> str:
        # No dimensions declared, so the critic's shape check cannot fire on a
        # query the oracle did not derive from a plan.
        return json.dumps(
            Plan(intent="aggregate", measure=None, rationale="oracle: replays the gold query").to_dict()
        )

    def _sql(self, payload: dict) -> str:
        sql = self.gold.get(str(payload.get("question", "")).strip().lower())
        return sql or "SELECT COUNT(*) AS answer FROM orders LIMIT 1"

    def _memo(self, payload: dict) -> str:
        columns = payload.get("columns") or []
        rows = payload.get("rows") or []
        if not rows or not columns:
            return json.dumps({"headline": "No rows returned.", "claims": [], "caveats": []})
        value_idx = len(columns) - 1
        label_idx = 0 if len(columns) > 1 else None
        lead = str(rows[0][label_idx]) if label_idx is not None else "overall"
        return json.dumps(
            {
                "headline": f"{len(rows)} row(s); {lead} leads with {rows[0][value_idx]}.",
                "claims": [
                    {"text": f"{lead} recorded {rows[0][value_idx]}.", "value": rows[0][value_idx],
                     "row": 0, "column": columns[value_idx]}
                ],
                "caveats": [],
            },
            default=str,
        )


CONTROLS = {"null": NullClient, "oracle": OracleClient}


def build_control(name: str, catalog: Catalog):
    return CONTROLS[name](catalog)
