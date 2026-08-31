"""Shared fixtures.

The warehouse is generated once per test session into a temporary directory, so
the suite is hermetic: it never reads or writes ``data/warehouse.db`` and it does
not care whether one exists. Generation is deterministic and takes about a
second, which is cheaper than the alternative of tests that pass only on a
machine where someone has already run the generator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.agents.graph import VantageAnalyst
from vantage.config import SETTINGS
from vantage.executor import ReadOnlyExecutor
from vantage.guardrails.sql_guard import SqlGuard
from vantage.llm.mock import MockAnalyst, QuestionParser
from vantage.retrieval.linker import SchemaLinker
from vantage.sql_compiler import SqlCompiler
from vantage.warehouse.catalog import build_catalog
from vantage.warehouse.generate import build


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("warehouse") / "test.db"
    build(path)
    return path


@pytest.fixture(scope="session")
def catalog(warehouse: Path):
    return build_catalog(warehouse)


@pytest.fixture(scope="session")
def linker(catalog):
    return SchemaLinker(catalog)


@pytest.fixture()
def guard(catalog):
    return SqlGuard(catalog, row_limit=500)


@pytest.fixture()
def executor(warehouse: Path):
    return ReadOnlyExecutor(warehouse, timeout_s=10, row_limit=500)


@pytest.fixture()
def compiler(catalog):
    return SqlCompiler(catalog)


@pytest.fixture(scope="session")
def parser(catalog):
    return QuestionParser(catalog)


@pytest.fixture()
def mock_client(catalog):
    return MockAnalyst(catalog)


@pytest.fixture(scope="session")
def settings(warehouse: Path):
    return SETTINGS.with_overrides(db_path=warehouse, model="mock")


@pytest.fixture(scope="session")
def analyst(settings) -> VantageAnalyst:
    return VantageAnalyst(settings=settings, log_runs=False)
