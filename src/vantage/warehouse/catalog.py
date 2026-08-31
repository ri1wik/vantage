"""Schema catalog: live introspection of the warehouse plus a curated glossary.

The catalog is what the schema linker searches over and what the SQL guardrails
validate against, so it is the single source of truth for "does this identifier
exist". Column descriptions and synonyms live in ``glossary.yaml`` next to this
module; everything else is read from the database itself.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

GLOSSARY_PATH = Path(__file__).parent / "glossary.yaml"


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    is_pk: bool
    description: str = ""
    synonyms: tuple[str, ...] = ()
    sample_values: tuple[str, ...] = ()

    def doc(self) -> str:
        parts = [self.name.replace("_", " "), self.description, " ".join(self.synonyms)]
        if self.sample_values:
            parts.append(" ".join(self.sample_values))
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...]
    row_count: int = 0
    description: str = ""
    synonyms: tuple[str, ...] = ()
    grain: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> Column | None:
        lowered = name.lower()
        for c in self.columns:
            if c.name.lower() == lowered:
                return c
        return None

    def doc(self) -> str:
        head = " ".join(
            p for p in [self.name.replace("_", " "), self.description, self.grain, " ".join(self.synonyms)] if p
        )
        return head + " " + " ".join(c.doc() for c in self.columns)

    def ddl(self) -> str:
        """Compact CREATE TABLE rendering handed to the SQL writer as context."""
        lines = [f"CREATE TABLE {self.name} ("]
        for c in self.columns:
            flag = " PRIMARY KEY" if c.is_pk else ""
            note = f"  -- {c.description}" if c.description else ""
            lines.append(f"    {c.name} {c.type}{flag},{note}")
        for fk in self.foreign_keys:
            lines.append(f"    FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column}),")
        body = "\n".join(lines).rstrip(",")
        return body + "\n);"


@dataclass
class Catalog:
    tables: dict[str, Table] = field(default_factory=dict)
    db_path: Path | None = None

    # -- lookups -----------------------------------------------------------
    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables))

    def table(self, name: str) -> Table | None:
        return self.tables.get(name.lower())

    def has_table(self, name: str) -> bool:
        return name.lower() in self.tables

    def has_column(self, table: str, column: str) -> bool:
        t = self.table(table)
        return bool(t and t.column(column))

    def resolve_column(self, column: str) -> list[str]:
        """Tables that own a column of this name (ambiguity check for the guard)."""
        return [t.name for t in self.tables.values() if t.column(column)]

    def total_rows(self) -> int:
        return sum(t.row_count for t in self.tables.values())

    # -- FK graph ----------------------------------------------------------
    def neighbours(self, table: str) -> set[str]:
        """Tables one join hop away, in either direction."""
        name = table.lower()
        out: set[str] = set()
        t = self.table(name)
        if t:
            out.update(fk.to_table for fk in t.foreign_keys)
        for other in self.tables.values():
            if any(fk.to_table == name for fk in other.foreign_keys):
                out.add(other.name)
        return out - {name}

    def fk_closure(self, seeds: set[str], hops: int = 1) -> set[str]:
        """Seed tables plus everything reachable within ``hops`` join hops."""
        frontier = {s.lower() for s in seeds if self.has_table(s)}
        seen = set(frontier)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for t in frontier:
                nxt |= self.neighbours(t)
            frontier = nxt - seen
            seen |= nxt
            if not frontier:
                break
        return seen

    def join_path(self, a: str, b: str) -> list[ForeignKey]:
        """Shortest FK path between two tables (breadth-first over the FK graph)."""
        a, b = a.lower(), b.lower()
        if a == b or not (self.has_table(a) and self.has_table(b)):
            return []
        queue: list[tuple[str, list[ForeignKey]]] = [(a, [])]
        seen = {a}
        while queue:
            node, path = queue.pop(0)
            for fk in self._edges(node):
                nxt = fk.to_table if fk.from_table == node else fk.from_table
                if nxt in seen:
                    continue
                new_path = path + [fk]
                if nxt == b:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
        return []

    def _edges(self, table: str) -> list[ForeignKey]:
        edges = list(self.table(table).foreign_keys) if self.table(table) else []
        for other in self.tables.values():
            edges.extend(fk for fk in other.foreign_keys if fk.to_table == table)
        return edges

    # -- rendering ---------------------------------------------------------
    def render(self, tables: list[str] | None = None) -> str:
        names = [t.lower() for t in (tables or self.table_names)]
        chosen = [self.tables[n] for n in names if n in self.tables]
        return "\n\n".join(t.ddl() for t in chosen)


def _load_glossary() -> dict:
    if not GLOSSARY_PATH.exists():  # pragma: no cover - glossary ships with the package
        return {}
    return yaml.safe_load(GLOSSARY_PATH.read_text()) or {}


def _sample_values(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> tuple[str, ...]:
    """Distinct values for low-cardinality text columns, used as retrieval signal."""
    if "CHAR" not in col_type.upper() and "TEXT" not in col_type.upper():
        return ()
    if column.endswith("_id") or column in {"email", "sku"}:
        return ()
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 26"
        ).fetchall()
    except sqlite3.Error:  # pragma: no cover - defensive
        return ()
    if len(rows) > 25:
        return ()
    return tuple(str(r[0]) for r in rows)


def build_catalog(db_path: Path) -> Catalog:
    """Introspect ``db_path`` and merge in the curated glossary."""
    glossary = _load_glossary()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    catalog = Catalog(db_path=db_path)

    table_names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in table_names:
        meta = glossary.get(name, {}) or {}
        col_meta = meta.get("columns", {}) or {}

        cols, pk = [], []
        for row in conn.execute(f"PRAGMA table_info({name})"):
            cm = col_meta.get(row["name"], {}) or {}
            if isinstance(cm, str):
                cm = {"description": cm}
            if row["pk"]:
                pk.append(row["name"])
            cols.append(
                Column(
                    name=row["name"],
                    type=row["type"],
                    nullable=not row["notnull"],
                    is_pk=bool(row["pk"]),
                    description=cm.get("description", ""),
                    synonyms=tuple(cm.get("synonyms", []) or []),
                    sample_values=_sample_values(conn, name, row["name"], row["type"]),
                )
            )

        fks = tuple(
            ForeignKey(name, r["from"], r["table"], r["to"] or "rowid")
            for r in conn.execute(f"PRAGMA foreign_key_list({name})")
        )
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]

        catalog.tables[name] = Table(
            name=name,
            columns=tuple(cols),
            primary_key=tuple(pk),
            foreign_keys=fks,
            row_count=count,
            description=meta.get("description", ""),
            synonyms=tuple(meta.get("synonyms", []) or []),
            grain=meta.get("grain", ""),
        )

    conn.close()
    return catalog


@lru_cache(maxsize=4)
def get_catalog(db_path: str) -> Catalog:
    """Cached catalog for a database path (catalogs are read-only and reusable)."""
    return build_catalog(Path(db_path))
