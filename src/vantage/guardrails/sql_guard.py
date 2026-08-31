"""AST guardrails: the only gate between a generated string and the database.

Every guarantee here is made on the parsed tree, never on a regex over the SQL
text, because string matching on SQL is defeated by comments, string literals and
whitespace. The guard is fail-closed: anything it cannot parse or cannot prove
safe is refused.

Guarantees, in order of application:

* the statement parses, and there is exactly one of them
* the root is a ``SELECT`` (optionally with CTEs, or a set operation over selects)
* no write, DDL, ``PRAGMA``, ``ATTACH`` or transaction node appears anywhere
* no filesystem or extension-loading function is called
* every table exists in the catalog, and (when a linked scope is given) is in scope
* every qualified column exists on the table it is qualified with
* a row ``LIMIT`` is present, injected when the model forgot one
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..warehouse.catalog import Catalog

DIALECT = "sqlite"

#: Node types that mutate data or schema. Presence anywhere in the tree is fatal.
WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Merge,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Set,
    exp.Grant,
)

#: SQLite functions that reach the filesystem or load native code.
BANNED_FUNCTIONS = frozenset({
    "load_extension",
    "readfile",
    "writefile",
    "edit",
    "fts3_tokenizer",
    "sqlite_compileoption_used",
    "sqlite_dbpage",
})

ERROR = "error"
WARN = "warn"


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: str = ERROR

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class GuardReport:
    """Outcome of guarding one candidate query."""

    ok: bool
    sql: str
    original_sql: str
    violations: list[Violation] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    limit_injected: bool = False

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == WARN]

    def summary(self) -> str:
        if self.ok and not self.violations:
            return "clean"
        return "; ".join(str(v) for v in self.violations) or "clean"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sql": self.sql,
            "tables": self.tables,
            "limit_injected": self.limit_injected,
            "violations": [{"code": v.code, "message": v.message, "severity": v.severity} for v in self.violations],
        }


def _strip_fences(sql: str) -> str:
    """Models like to wrap SQL in markdown fences. Take the fenced body if present."""
    text = sql.strip()
    if not text.startswith("```"):
        return text
    body = text.split("```")
    if len(body) < 2:
        return text
    inner = body[1]
    if inner.lower().startswith("sql"):
        inner = inner[3:]
    return inner.strip()


class SqlGuard:
    """Validates and, where safe, rewrites a candidate SELECT."""

    def __init__(self, catalog: Catalog, row_limit: int = 500) -> None:
        self.catalog = catalog
        self.row_limit = row_limit

    # -- public API --------------------------------------------------------
    def check(self, sql: str, allowed_tables: list[str] | None = None) -> GuardReport:
        original = sql
        text = _strip_fences(sql)
        violations: list[Violation] = []

        if not text.strip():
            return GuardReport(False, "", original, [Violation("EMPTY_SQL", "no SQL was produced")])

        try:
            parsed = sqlglot.parse(text, read=DIALECT)
        except Exception as err:  # sqlglot raises several parse error types
            return GuardReport(
                False, "", original, [Violation("PARSE_ERROR", f"could not parse SQL: {err}")]
            )
        # A trailing semicolon or comment parses to an empty node; that is not a
        # second statement, so drop it before the stacked-query check.
        statements = [
            st for st in parsed if st is not None and st.sql(dialect=DIALECT, comments=False).strip()
        ]

        if len(statements) != 1:
            return GuardReport(
                False,
                "",
                original,
                [Violation("MULTIPLE_STATEMENTS", f"expected exactly one statement, found {len(statements)}")],
            )

        tree = statements[0]

        violations += self._check_read_only(tree)
        violations += self._check_functions(tree)
        tables = self._referenced_tables(tree)
        violations += self._check_tables(tables, allowed_tables)
        columns, column_violations = self._check_columns(tree)
        violations += column_violations

        if any(v.severity == ERROR for v in violations):
            return GuardReport(False, "", original, violations, sorted(tables), sorted(columns))

        guarded, limit_action = self._enforce_limit(tree)
        if limit_action == "injected":
            violations.append(
                Violation("LIMIT_INJECTED", f"no LIMIT present; capped at {self.row_limit} rows", WARN)
            )
        elif limit_action == "clamped":
            violations.append(
                Violation("LIMIT_CLAMPED", f"requested LIMIT exceeded the cap; reduced to {self.row_limit}", WARN)
            )

        return GuardReport(
            ok=True,
            sql=guarded.sql(dialect=DIALECT, pretty=True),
            original_sql=original,
            violations=violations,
            tables=sorted(tables),
            columns=sorted(columns),
            limit_injected=limit_action != "kept",
        )

    def is_safe(self, sql: str, allowed_tables: list[str] | None = None) -> bool:
        return self.check(sql, allowed_tables).ok

    # -- individual passes -------------------------------------------------
    def _check_read_only(self, tree: exp.Expression) -> list[Violation]:
        out: list[Violation] = []
        for node in tree.find_all(*WRITE_NODES):
            out.append(
                Violation(
                    "WRITE_OPERATION",
                    f"{type(node).__name__.upper()} is not allowed; Vantage is read-only",
                )
            )
        if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
            if not out:  # a write node already explains the refusal
                out.append(
                    Violation("NON_SELECT", f"root statement is {type(tree).__name__}, expected SELECT")
                )
        return out

    def _check_functions(self, tree: exp.Expression) -> list[Violation]:
        out: list[Violation] = []
        for node in tree.find_all(exp.Anonymous):
            name = (node.this or "").lower() if isinstance(node.this, str) else ""
            if name in BANNED_FUNCTIONS:
                out.append(Violation("BANNED_FUNCTION", f"function {name}() is not permitted"))
        for node in tree.find_all(exp.Func):
            name = getattr(node, "sql_name", lambda: "")()
            if isinstance(name, str) and name.lower() in BANNED_FUNCTIONS:
                out.append(Violation("BANNED_FUNCTION", f"function {name.lower()}() is not permitted"))
        return out

    def _cte_names(self, tree: exp.Expression) -> set[str]:
        return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    def _referenced_tables(self, tree: exp.Expression) -> set[str]:
        ctes = self._cte_names(tree)
        names = set()
        for node in tree.find_all(exp.Table):
            name = (node.name or "").lower()
            if name and name not in ctes:
                names.add(name)
        return names

    def _check_tables(self, tables: set[str], allowed: list[str] | None) -> list[Violation]:
        out: list[Violation] = []
        for name in sorted(tables):
            if not self.catalog.has_table(name):
                known = ", ".join(self.catalog.table_names)
                out.append(
                    Violation("UNKNOWN_TABLE", f"table '{name}' does not exist. Known tables: {known}")
                )
        if allowed is not None:
            scope = {t.lower() for t in allowed}
            for name in sorted(tables):
                if self.catalog.has_table(name) and name not in scope:
                    out.append(
                        Violation(
                            "OUT_OF_SCOPE_TABLE",
                            f"table '{name}' was not in the linked schema {sorted(scope)}",
                            WARN,
                        )
                    )
        return out

    def _alias_map(self, tree: exp.Expression) -> dict[str, str]:
        """Map every table alias (and bare table name) to the real table it means."""
        mapping: dict[str, str] = {}
        for node in tree.find_all(exp.Table):
            real = (node.name or "").lower()
            mapping[real] = real
            alias = (node.alias or "").lower()
            if alias:
                mapping[alias] = real
        return mapping

    def _opaque_sources(self, tree: exp.Expression) -> set[str]:
        """Aliases whose columns Vantage cannot resolve: CTEs and derived tables."""
        opaque = self._cte_names(tree)
        for node in tree.find_all(exp.Subquery):
            alias = (node.alias or "").lower()
            if alias:
                opaque.add(alias)
        return opaque

    def _select_aliases(self, tree: exp.Expression) -> set[str]:
        return {
            (a.alias or "").lower()
            for a in tree.find_all(exp.Alias)
            if a.alias
        }

    def _check_columns(self, tree: exp.Expression) -> tuple[set[str], list[Violation]]:
        aliases = self._alias_map(tree)
        opaque = self._opaque_sources(tree)
        select_aliases = self._select_aliases(tree)
        real_tables = {t for t in self._referenced_tables(tree) if self.catalog.has_table(t)}

        seen: set[str] = set()
        out: list[Violation] = []
        for node in tree.find_all(exp.Column):
            col = (node.name or "").lower()
            if not col or col == "*":
                continue
            qualifier = (node.table or "").lower()
            seen.add(f"{qualifier}.{col}" if qualifier else col)

            if qualifier:
                if qualifier in opaque:
                    continue
                target = aliases.get(qualifier, qualifier)
                if not self.catalog.has_table(target):
                    continue  # already reported as UNKNOWN_TABLE
                if not self.catalog.has_column(target, col):
                    near = ", ".join(self.catalog.table(target).column_names)
                    out.append(
                        Violation(
                            "UNKNOWN_COLUMN",
                            f"column '{col}' does not exist on '{target}'. Columns are: {near}",
                        )
                    )
                continue

            # Unqualified: accept a projection alias, or membership in any table in scope.
            if col in select_aliases or opaque:
                continue
            if not real_tables:
                out.append(
                    Violation(
                        "UNKNOWN_COLUMN",
                        f"column '{col}' is referenced but the query selects from no table",
                    )
                )
            elif not any(self.catalog.has_column(t, col) for t in real_tables):
                out.append(
                    Violation(
                        "UNKNOWN_COLUMN",
                        f"column '{col}' does not exist on any table in scope {sorted(real_tables)}",
                    )
                )
        return seen, out

    def _enforce_limit(self, tree: exp.Expression) -> tuple[exp.Expression, str]:
        """Cap the outermost query. Returns "kept", "clamped" or "injected"."""
        existing = tree.args.get("limit")
        if existing is not None:
            value = existing.expression
            if isinstance(value, exp.Literal) and value.is_int and int(value.name) > self.row_limit:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(self.row_limit)))
                return tree, "clamped"
            return tree, "kept"
        if isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
            return tree.limit(self.row_limit), "injected"
        return tree, "kept"
