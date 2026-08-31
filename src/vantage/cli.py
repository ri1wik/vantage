"""Command line interface.

    vantage build                       # generate the 258k-row warehouse
    vantage ask "revenue by category in 2024"
    vantage ask "..." --trace           # plan, linked tables and every attempt
    vantage sql "SELECT ..."            # guard a query, then run it
    vantage schema orders
    vantage bench --model mock
    vantage serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .config import SETTINGS

console = Console()


def _print_rows(columns: list[str], rows: list[list], limit: int = 25) -> None:
    if not columns:
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows[:limit]:
        table.add_row(*("NULL" if c is None else str(c) for c in row))
    console.print(table)
    if len(rows) > limit:
        console.print(f"[dim]... {len(rows) - limit} more row(s)[/dim]")


def cmd_build(args: argparse.Namespace) -> int:
    from .warehouse.generate import main as generate

    return generate(["--out", args.out or str(SETTINGS.db_path)])


def cmd_ask(args: argparse.Namespace) -> int:
    from .agents.graph import VantageAnalyst

    settings = SETTINGS.with_overrides(model=args.model) if args.model else SETTINGS
    analyst = VantageAnalyst(settings=settings, log_runs=not args.no_log)
    answer = analyst.ask(args.question, max_attempts=args.max_attempts)

    if args.json:
        print(json.dumps(answer.as_dict(), indent=2, default=str))
        return 0 if answer.status == "answered" else 1

    if answer.status == "refused":
        refusal = answer.refusal or {}
        console.print(f"[bold yellow]Refused[/bold yellow] ({refusal.get('category')})")
        console.print(refusal.get("reason", ""))
        if refusal.get("suggestion"):
            console.print(f"[dim]{refusal['suggestion']}[/dim]")
        return 2

    if answer.status != "answered":
        console.print(f"[bold red]Failed[/bold red]: {answer.error or 'no answer produced'}")
        for attempt in answer.attempts:
            console.print(f"  attempt {attempt['n']}: {attempt['critique']}")
        return 1

    console.print(Syntax(answer.sql or "", "sql", theme="ansi_dark", word_wrap=True))
    console.print()
    _print_rows(answer.columns, answer.rows, limit=args.rows)
    console.print()
    console.print(answer.memo_text)
    console.print(
        f"\n[dim]{answer.attempt_count} attempt(s) | {answer.latency_ms:.0f}ms | "
        f"faithfulness {answer.faithfulness:.0%} | trace {answer.trace_id}[/dim]"
    )

    if args.trace:
        console.print("\n[bold]Plan[/bold]")
        console.print(json.dumps(answer.plan, indent=2))
        console.print("\n[bold]Linked schema[/bold]")
        linked = answer.linked or {}
        console.print(f"  seeds:   {linked.get('seeds')}")
        console.print(f"  bridges: {linked.get('bridges')}")
        console.print(f"  scope:   {linked.get('scope')}")
        console.print(f"  recall:  {linked.get('recall')}")
        console.print("\n[bold]Attempts[/bold]")
        for attempt in answer.attempts:
            console.print(
                f"  #{attempt['n']} guard_ok={attempt['guard_ok']} verdict={attempt['verdict']}"
            )
            if attempt["guard_violations"]:
                for violation in attempt["guard_violations"]:
                    console.print(f"      {violation}")
            if attempt["repair_hint"]:
                console.print(f"      hint: {attempt['repair_hint']}")
        console.print("\n[bold]Facts check[/bold]")
        console.print(json.dumps(answer.fact_check, indent=2))
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    from .executor import ReadOnlyExecutor
    from .guardrails.sql_guard import SqlGuard
    from .warehouse.catalog import get_catalog

    catalog = get_catalog(str(SETTINGS.db_path))
    report = SqlGuard(catalog, row_limit=SETTINGS.row_limit).check(args.sql)
    for violation in report.violations:
        style = "red" if violation.severity == "error" else "yellow"
        console.print(f"[{style}]{violation}[/{style}]")
    if not report.ok:
        return 1
    if args.check_only:
        console.print("[green]guard passed[/green]")
        return 0

    result, error = ReadOnlyExecutor(
        SETTINGS.db_path, SETTINGS.query_timeout_s, SETTINGS.row_limit
    ).try_run(report.sql)
    if error is not None:
        console.print(f"[red]{error.kind}: {error}[/red]")
        return 1
    _print_rows(result.columns, [list(r) for r in result.rows], limit=args.rows)
    console.print(f"[dim]{result.row_count} row(s) in {result.elapsed_ms:.0f}ms[/dim]")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from .warehouse.catalog import get_catalog

    catalog = get_catalog(str(SETTINGS.db_path))
    if args.table:
        table = catalog.table(args.table)
        if table is None:
            console.print(f"[red]unknown table '{args.table}'[/red]")
            console.print(f"known: {', '.join(catalog.table_names)}")
            return 1
        console.print(Syntax(table.ddl(), "sql", theme="ansi_dark"))
        return 0

    table_view = Table(show_header=True, header_style="bold", box=None)
    for column in ("table", "rows", "grain", "description"):
        table_view.add_column(column, overflow="fold")
    for name in catalog.table_names:
        t = catalog.tables[name]
        table_view.add_row(t.name, f"{t.row_count:,}", t.grain, t.description)
    console.print(table_view)
    console.print(f"[dim]{catalog.total_rows():,} rows across {len(catalog.tables)} tables[/dim]")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bench.runner import main as bench_main

    argv = ["--model", args.model]
    if args.out:
        argv += ["--out", args.out]
    if args.tier:
        for tier in args.tier:
            argv += ["--tier", tier]
    if args.fail_under is not None:
        argv += ["--fail-under", str(args.fail_under)]
    return bench_main(argv)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("vantage.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .run_log import RunLogger

    record = RunLogger(SETTINGS.log_dir).read(args.trace_id)
    if record is None:
        console.print(f"[red]no run logged with trace id {args.trace_id}[/red]")
        return 1
    print(json.dumps(record, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vantage", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="generate the demo warehouse")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--model", default=None, help="mock | openai | groq | gemini | ollama")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--rows", type=int, default=25, help="rows to display")
    p.add_argument("--trace", action="store_true", help="show plan, linked tables and attempts")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("sql", help="guard and run a SELECT")
    p.add_argument("sql")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--rows", type=int, default=25)
    p.set_defaults(func=cmd_sql)

    p = sub.add_parser("schema", help="show the warehouse schema")
    p.add_argument("table", nargs="?", default=None)
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("bench", help="run vantage-bench")
    p.add_argument("--model", default="mock")
    p.add_argument("--out", default=None)
    p.add_argument("--tier", action="append", default=None)
    p.add_argument("--fail-under", type=float, default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("serve", help="run the FastAPI app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("trace", help="print a logged run by trace id")
    p.add_argument("trace_id")
    p.set_defaults(func=cmd_trace)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
