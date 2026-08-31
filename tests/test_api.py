"""The HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, settings):
    from vantage import api
    from vantage.agents.graph import VantageAnalyst

    api.get_analyst.cache_clear()
    monkeypatch.setattr(api, "SETTINGS", settings)
    monkeypatch.setattr(
        api, "get_analyst", lambda: VantageAnalyst(settings=settings, log_runs=False)
    )
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_the_warehouse_and_model(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["rows"] == 258_000
    assert body["tables"] == 10


def test_ask_returns_the_answer_and_the_whole_trace(client) -> None:
    body = client.post("/ask", json={"question": "Revenue by sales channel"}).json()
    assert body["status"] == "answered"
    assert body["columns"] == ["channel", "revenue"]
    assert body["attempt_count"] == 1
    assert body["plan"] and body["linked"] and body["fact_check"]


def test_sql_validate_refuses_writes_without_executing(client) -> None:
    body = client.post("/sql/validate", json={"sql": "DELETE FROM orders"}).json()
    assert body["ok"] is False
    assert body["violations"][0]["code"] == "WRITE_OPERATION"


def test_schema_endpoint_exposes_every_table_with_its_foreign_keys(client) -> None:
    body = client.get("/schema").json()
    assert body["total_rows"] == 258_000
    assert len(body["tables"]) == 10
    order_items = next(t for t in body["tables"] if t["name"] == "order_items")
    assert {fk["references"] for fk in order_items["foreign_keys"]} == {
        "orders.order_id",
        "products.product_id",
    }
