"""Rendering for bench results: a console block and a markdown report."""

from __future__ import annotations

from typing import Any

TIER_LABELS = {
    "semantic_accuracy": "Tier 1  semantic execution accuracy",
    "self_correction": "Tier 2  self-correction recovery",
    "refusal": "Tier 3  refusal correctness",
    "memo_faithfulness": "Tier 4  memo faithfulness",
}


def _bar(rate: float, width: int = 20) -> str:
    filled = int(round(rate * width))
    return "#" * filled + "." * (width - filled)


def render_console(results: dict[str, Any]) -> str:
    s = results["summary"]
    lines = [
        f"vantage-bench  model={results['model']}  warehouse={results['warehouse_rows']:,} rows  "
        f"{results['duration_s']}s",
        "=" * 78,
    ]
    for tier, label in TIER_LABELS.items():
        t = s["tiers"].get(tier)
        if not t:
            continue
        lines.append(f"{label:38} {t['passed']:>2}/{t['total']:<2} {_bar(t['rate'])} {t['rate']:>6.1%}")
    lines += [
        "-" * 78,
        f"{'overall':38} {s['passed']:>2}/{s['cases']:<2} {_bar(s['pass_rate'])} {s['pass_rate']:>6.1%}",
        "",
        f"  refusal precision/recall/F1   {s['refusal']['precision']:.2f} / "
        f"{s['refusal']['recall']:.2f} / {s['refusal']['f1']:.2f}"
        f"   (category accuracy {s['refusal']['category_accuracy']:.0%})",
        f"  self-correction recovery      {s['self_correction']['recovery_rate']:.1%} over "
        f"{s['self_correction']['total']} "
        f"{'injected faults' if results.get('controls_injected', True) else 'cases (no fault injected)'}, "
        f"{s['self_correction']['mean_attempts']} attempts on average",
        f"  memo faithfulness (mean)      {s['memo_faithfulness']['mean']:.1%} "
        f"across {s['memo_faithfulness']['checked']} memos",
        f"  schema linker recall (mean)   {s['linker_recall']['mean']:.1%} "
        f"({s['linker_recall']['perfect']}/{s['linker_recall']['measured']} cases at 100%)",
        f"  latency p50 / p95             {s['latency_ms']['p50']:.0f}ms / {s['latency_ms']['p95']:.0f}ms",
    ]
    if not results.get("controls_injected", True):
        lines += [
            "",
            "  note: fault injection and the unfaithful-memo plant apply to the deterministic",
            "        baseline only. Tiers 2 and 4 here measure this model's own errors.",
        ]
    failures = [c for c in results["cases"] if not c["passed"]]
    if failures:
        lines += ["", f"  {len(failures)} failing case(s):"]
        lines += [f"    {c['id']:8} {c['reason'][:88]}" for c in failures]
    return "\n".join(lines)


def render_markdown(results: dict[str, Any]) -> str:
    s = results["summary"]
    lines = [
        "# vantage-bench report",
        "",
        f"- **model**: `{results['model']}`"
        + (f" (`{results['model_name']}`)" if results.get("model_name") else ""),
        f"- **warehouse**: {results['warehouse_rows']:,} rows at `{results['database']}`",
        f"- **attempt budget**: {results['max_attempts']}",
        f"- **duration**: {results['duration_s']}s on Python {results['python']}",
        f"- **controls injected**: {'yes' if results.get('controls_injected', True) else 'no (baseline-only)'}",
        "",
        f"## Overall: {s['passed']}/{s['cases']} ({s['pass_rate']:.1%})",
        "",
        "| Tier | Passed | Rate |",
        "| --- | ---: | ---: |",
    ]
    for tier, label in TIER_LABELS.items():
        t = s["tiers"].get(tier)
        if t:
            lines.append(f"| {label} | {t['passed']}/{t['total']} | {t['rate']:.1%} |")

    lines += [
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Semantic execution accuracy | {s['semantic_execution_accuracy']:.1%} |",
        f"| Self-correction recovery | {s['self_correction']['recovery_rate']:.1%} |",
        f"| Mean attempts under fault | {s['self_correction']['mean_attempts']} |",
        f"| Refusal precision | {s['refusal']['precision']:.2f} |",
        f"| Refusal recall | {s['refusal']['recall']:.2f} |",
        f"| Refusal F1 | {s['refusal']['f1']:.2f} |",
        f"| Refusal category accuracy | {s['refusal']['category_accuracy']:.1%} |",
        f"| Memo faithfulness (mean) | {s['memo_faithfulness']['mean']:.1%} |",
        f"| Schema linker recall (mean) | {s['linker_recall']['mean']:.1%} |",
        f"| Latency p50 | {s['latency_ms']['p50']:.0f} ms |",
        f"| Latency p95 | {s['latency_ms']['p95']:.0f} ms |",
        "",
        "## Cases",
        "",
        "| Case | Tier | Result | Attempts | Latency | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        mark = "pass" if case["passed"] else "**fail**"
        lines.append(
            f"| `{case['id']}` | {case['tier']} | {mark} | {case['attempts']} | "
            f"{case['latency_ms']:.0f} ms | {case['reason'][:110]} |"
        )
    return "\n".join(lines) + "\n"
