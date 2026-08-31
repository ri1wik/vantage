"""vantage-bench runner.

    python -m bench.runner --model mock --out bench/results

Runs every case in ``cases.yaml`` through the real graph: the same planner,
linker, guardrails, executor, critic and facts checker the API uses. Nothing is
stubbed, so a bench pass is evidence about the system rather than about the
harness.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from vantage.agents.graph import Answer, VantageAnalyst
from vantage.config import SETTINGS
from vantage.executor import ReadOnlyExecutor

from .controls import CONTROLS, build_control
from .metrics import CaseScore, answers_match, summarize
from .report import render_markdown, render_console

CASES_PATH = Path(__file__).parent / "cases.yaml"
TIERS = ("semantic_accuracy", "self_correction", "refusal", "memo_faithfulness")


def load_cases(path: Path = CASES_PATH, tiers: list[str] | None = None) -> list[dict]:
    cases = yaml.safe_load(path.read_text()) or []
    if tiers:
        cases = [c for c in cases if c["tier"] in tiers]
    return cases


class BenchRunner:
    def __init__(
        self,
        model: str = "mock",
        db_path: Path | None = None,
        max_attempts: int = 3,
        model_name: str | None = None,
    ) -> None:
        self.settings = SETTINGS.with_overrides(
            model=model, max_attempts=max_attempts, model_name=model_name
        )
        if db_path:
            self.settings = self.settings.with_overrides(db_path=Path(db_path).resolve())
        self.model = model
        #: Fault injection and the unfaithful-memo plant are properties of the
        #: deterministic baseline. Against any other model the tiers measure the
        #: model's own errors, so the controls are switched off rather than faked.
        self.supports_controls = model == "mock"
        self.executor = ReadOnlyExecutor(
            self.settings.db_path, self.settings.query_timeout_s, self.settings.row_limit
        )
        # One analyst per (fault, unfaithful) combination; building the graph and
        # the TF-IDF index is the expensive part and it is safe to share.
        self._analysts: dict[tuple[str | None, bool], VantageAnalyst] = {}

    def analyst(self, fault: str | None = None, unfaithful: bool = False) -> VantageAnalyst:
        key = (fault, unfaithful)
        if key not in self._analysts:
            if self.model in CONTROLS:
                from vantage.warehouse.catalog import get_catalog

                client = build_control(self.model, get_catalog(str(self.settings.db_path)))
                self._analysts[key] = VantageAnalyst(
                    settings=self.settings, client=client, log_runs=False
                )
            else:
                kwargs: dict[str, Any] = (
                    {"fault_profile": fault, "unfaithful_memo": unfaithful}
                    if self.supports_controls
                    else {}
                )
                self._analysts[key] = VantageAnalyst(settings=self.settings, log_runs=False, **kwargs)
        return self._analysts[key]

    # -- per tier ----------------------------------------------------------
    def score_case(self, case: dict) -> CaseScore:
        tier = case["tier"]
        handler = {
            "semantic_accuracy": self._score_semantic,
            "self_correction": self._score_correction,
            "refusal": self._score_refusal,
            "memo_faithfulness": self._score_memo,
        }[tier]
        return handler(case)

    def _run(self, case: dict, fault: str | None = None, unfaithful: bool = False) -> Answer:
        analyst = self.analyst(fault, unfaithful)
        return analyst.ask(case["question"], max_attempts=case.get("max_attempts"))

    def _linker_recall(self, answer: Answer, case: dict) -> float | None:
        gold = case.get("gold_tables")
        if not gold or not answer.linked:
            return None
        linked = set(answer.linked.get("tables", []))
        return len([t for t in gold if t in linked]) / len(gold)

    def _base(self, case: dict, answer: Answer, passed: bool, reason: str) -> CaseScore:
        return CaseScore(
            id=case["id"],
            tier=case["tier"],
            question=case["question"],
            passed=passed,
            reason=reason,
            status=answer.status,
            attempts=answer.attempt_count,
            self_corrected=answer.self_corrected,
            latency_ms=answer.latency_ms,
            linker_recall=self._linker_recall(answer, case),
            faithfulness=answer.faithfulness if answer.status == "answered" else None,
            sql=answer.sql or "",
        )

    def _compare_to_gold(self, case: dict, answer: Answer) -> tuple[bool, str]:
        if answer.status != "answered":
            detail = (answer.refusal or {}).get("reason") or answer.error or "no answer produced"
            return False, f"status={answer.status}: {detail}"
        gold = self.executor.run(case["gold_sql"])
        ordered = "ORDER BY" in case["gold_sql"].upper() and "LIMIT" in case["gold_sql"].upper()
        return answers_match(gold, answer.columns, answer.rows, ordered=ordered)

    def _score_semantic(self, case: dict) -> CaseScore:
        answer = self._run(case)
        passed, reason = self._compare_to_gold(case, answer)
        return self._base(case, answer, passed, reason or "answer matches gold")

    def _score_correction(self, case: dict) -> CaseScore:
        answer = self._run(case, fault=case.get("fault"))
        matched, reason = self._compare_to_gold(case, answer)
        diagnosed = any(a.get("verdict") == "repair" for a in answer.attempts)
        if not self.supports_controls:
            # No fault could be injected, so the tier degrades to "was the answer
            # right", and a repair only counts if the model needed one.
            passed = matched
            reason = reason or ("answered correctly" + (" after a repair" if diagnosed else " first time"))
            score = self._base(case, answer, passed, reason)
            score.detail = {"fault": None, "diagnosed": diagnosed, "injected": False,
                            "answer_matches_gold": matched}
            return score
        passed = bool(matched and diagnosed and answer.attempt_count > 1)
        if matched and not diagnosed:
            reason = "answer correct but no fault was diagnosed; the repair loop was not exercised"
        elif matched and answer.attempt_count <= 1:
            reason = "answer correct on the first attempt; the injected fault did not take"
        score = self._base(case, answer, passed, reason or "fault diagnosed and repaired")
        score.detail = {
            "fault": case.get("fault"),
            "injected": True,
            "diagnosed": diagnosed,
            "first_verdict": answer.attempts[0]["verdict"] if answer.attempts else "",
            "answer_matches_gold": matched,
        }
        return score

    def _score_refusal(self, case: dict) -> CaseScore:
        expected = bool(case.get("expect_refusal"))
        answer = self._run(case)
        refused = answer.status == "refused"
        category = (answer.refusal or {}).get("category", "")
        category_match = (not expected) or category == case.get("refusal_category")

        if expected and refused and category_match:
            passed, reason = True, f"refused as {category}"
        elif expected and refused:
            passed, reason = False, f"refused, but as '{category}' instead of '{case.get('refusal_category')}'"
        elif expected:
            passed, reason = False, "should have refused but produced an answer"
        elif refused:
            passed, reason = False, f"wrongly refused an answerable question as '{category}'"
        else:
            passed, reason = self._compare_to_gold(case, answer)
            reason = reason or "answered correctly, as expected"

        score = self._base(case, answer, passed, reason)
        score.detail = {
            "expected_refusal": expected,
            "refused": refused,
            "category": category,
            "expected_category": case.get("refusal_category"),
            "category_match": category_match,
        }
        return score

    def _score_memo(self, case: dict) -> CaseScore:
        control = bool(case.get("unfaithful_control")) and self.supports_controls
        answer = self._run(case, unfaithful=control)
        check = answer.fact_check or {}
        unverified = list(check.get("unverified", []))

        if answer.status != "answered":
            passed, reason = False, f"status={answer.status}"
        elif control:
            # The control passes when the checker caught the planted figure AND
            # it never reached the rendered memo.
            leaked = any(u.lstrip("-+$").rstrip("%").replace(",", "") in answer.memo_text for u in unverified)
            passed = bool(unverified) and not leaked
            reason = (
                "planted figure caught and stripped"
                if passed
                else ("planted figure was not detected" if not unverified else "planted figure leaked into the memo")
            )
        else:
            passed = not unverified
            reason = "every figure traced to the result set" if passed else f"unverified: {unverified}"

        score = self._base(case, answer, passed, reason)
        score.detail = {
            "unfaithful_control": control,
            "unverified": unverified,
            "checked": check.get("checked", 0),
            "memo": answer.memo_text,
        }
        return score

    # -- driver ------------------------------------------------------------
    def run(self, cases: list[dict], on_case=None) -> dict[str, Any]:
        started = time.time()
        scores: list[CaseScore] = []
        for case in cases:
            try:
                score = self.score_case(case)
            except Exception as err:  # a harness crash is a failed case, not a lost run
                score = CaseScore(
                    id=case["id"],
                    tier=case["tier"],
                    question=case["question"],
                    passed=False,
                    reason=f"harness error: {type(err).__name__}: {err}",
                    status="error",
                )
            scores.append(score)
            if on_case:
                on_case(score)

        return {
            "model": self.model,
            "model_name": self.settings.model_name,
            "controls_injected": self.supports_controls,
            "database": str(self.settings.db_path),
            "warehouse_rows": self._warehouse_rows(),
            "max_attempts": self.settings.max_attempts,
            "python": platform.python_version(),
            "duration_s": round(time.time() - started, 2),
            "summary": summarize(scores),
            "cases": [s.as_dict() for s in scores],
        }

    def _warehouse_rows(self) -> int:
        from vantage.warehouse.catalog import get_catalog

        return get_catalog(str(self.settings.db_path)).total_rows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run vantage-bench.")
    ap.add_argument(
        "--model",
        default="mock",
        help="mock | openai | groq | gemini | ollama | null | oracle "
        "(null and oracle are harness controls: null must score near zero, "
        "oracle bounds the answer-correctness tiers)",
    )
    ap.add_argument("--model-name", default=None, help="override the provider's default model id")
    ap.add_argument("--db", default=None, help="warehouse path (default: VANTAGE_DB)")
    ap.add_argument("--tier", action="append", choices=TIERS, help="restrict to a tier (repeatable)")
    ap.add_argument("--case", action="append", help="run specific case ids")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", default=None, help="directory for results.json and report.md")
    ap.add_argument("--fail-under", type=float, default=None, help="exit 1 if pass rate is below this")
    ap.add_argument("--quiet", action="store_true", help="suppress per-case lines; still prints the summary")
    args = ap.parse_args(argv)

    cases = load_cases(tiers=args.tier)
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    runner = BenchRunner(
        model=args.model, db_path=args.db, max_attempts=args.max_attempts, model_name=args.model_name
    )

    def progress(score: CaseScore) -> None:
        if not args.quiet:
            mark = "PASS" if score.passed else "FAIL"
            print(f"  {mark}  {score.id:8} {score.question[:56]:58} {score.reason[:60]}")

    if not args.quiet:
        print(f"vantage-bench | model={args.model} | {len(cases)} case(s)\n")
    results = runner.run(cases, on_case=progress)

    print()
    print(render_console(results))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        (out_dir / "report.md").write_text(render_markdown(results))
        print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'report.md'}")

    rate = results["summary"]["pass_rate"]
    if args.fail_under is not None and rate < args.fail_under:
        print(f"\npass rate {rate:.1%} is below the --fail-under floor of {args.fail_under:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
