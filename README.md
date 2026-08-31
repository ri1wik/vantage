# Vantage

**A self-correcting multi-agent data analyst.** Vantage turns a plain-English
question into AST-guarded, read-only SQL over a 258,000-row warehouse, repairs
its own failed queries inside a bounded loop, and refuses to state a number it
cannot trace back to a row it returned.

```
$ vantage ask "Which carrier has the most lost shipments?"

SELECT
  s.carrier AS carrier,
  COUNT(s.shipment_id) AS shipment_count
FROM shipments AS s
WHERE s.status = 'lost'
GROUP BY 1
ORDER BY shipment_count DESC
LIMIT 1

carrier  shipment_count
DHL      126

1 row(s) returned; DHL leads on shipment count with 126.
  - DHL recorded 126 for shipment count.
Caveats:
  * Scope: lost shipments only.

1 attempt(s) | 6ms | faithfulness 100% | trace 4f2a91c0b7de
```

---

## Why this exists

Text-to-SQL demos fail in three ways that a demo never shows you.

1. **The query is wrong in a way that runs.** A missing `GROUP BY` makes SQLite
   return one arbitrary row. It parses, it executes, the shape looks right, and
   the number is garbage.
2. **The schema is hallucinated.** `orders.total_amount` does not exist. The
   model is confident anyway.
3. **The prose is invented.** The query is right, the rows are right, and the
   summary says "roughly a third of revenue" because that sounded plausible.

Vantage is built around controls for each: an AST guard that validates against
the live catalog before anything executes, a critic that reads the parse tree
rather than the row count, and a facts checker that strips any figure the result
set does not support. `vantage-bench` measures whether those controls actually
work, and ships a null-agent control so the benchmark cannot flatter itself.

---

## Architecture

```
                         +-------------------------------------+
  question ------------> |  planner       intent, measure,     |
                         |                dimensions, refusal  |
                         +------------------+------------------+
                                            |  refuse -> stop
                                            v
                         +-------------------------------------+
                         |  schema linker  TF-IDF + lexicon +  |
                         |                 FK closure + anchors|
                         +------------------+------------------+
                                            v
    +------------------>  +-------------------------------------+
    |                     |  SQL writer -> sqlglot AST guard    |
    |                     +------------------+------------------+
    |                                        v
    |                     +-------------------------------------+
    |                     |  executor       read-only, authorizer|
    |                     |                 + timeout + row cap  |
    |                     +------------------+------------------+
    |                                        v
    |   repair            +-------------------------------------+
    +-------------------  |  critic         guard / execution /  |
        (bounded)         |                 AST shape verdict    |
                          +------------------+------------------+
                                             |  accept
                                             v
                          +-------------------------------------+
                          |  memo composer  -> facts checker    |
                          |                    strips unverified|
                          +-------------------------------------+
```

Five agents on a LangGraph state machine, plus two nodes that never call a model
(the executor, and the critic, which is deliberately rule-based).

### The agents

| Agent | Job | Why it is separate |
| --- | --- | --- |
| **planner** | Decides the analysis: measure, dimensions, filters, ordering, or a refusal with a category. | Refusal is a planning decision. Deciding it before any schema is retrieved means an out-of-scope question never reaches a SQL writer that would try to answer it. |
| **schema linker** | Retrieves the smallest join-connected table set that can answer the question. | Handing the model the whole schema is the cheapest way to get a hallucinated join. Handing it too little makes an answerable question unanswerable. |
| **SQL writer** | Emits one SELECT for the plan, then submits it to the AST guard. | The writer never touches the database. Everything it produces is a candidate until the guard clears it. |
| **critic** | Reads the guard report, the execution outcome and the parse tree, then decides accept, repair or abandon. | Its job is to be right about mechanical facts. A rule is cheaper and more reliable than a model at "did this parse, did it run, does the shape match the plan". |
| **memo composer** | Writes the summary, which is then checked figure by figure. | Separating composition from verification means the composer cannot mark its own homework. |

### Hallucination control, end to end

**1. Hybrid schema retrieval (100% linker recall on the bench).** Four passes,
each covering a failure of the one before it:

- A **lexicon pass** pins the fact table implied by the measure before any
  statistical scoring runs, so "revenue" always means `order_items.line_total`.
- A **TF-IDF pass** scores every table document (name, description, grain,
  synonyms, column docs, low-cardinality sample values) against the question.
- An **FK-closure pass** adds only the bridge tables that sit on the shortest
  join path between two selected tables, so joins are always spellable.
- A **column-ownership pass** and a **temporal-anchor pass** catch the two cases
  statistics is worst at. "Revenue by currency" names a column that lives only on
  `orders`, a table TF-IDF buries under `revenue`. And `order_items` carries the
  revenue but no date, so any time-grained revenue question also needs `orders`;
  rather than hardcode that pair, the linker adds the nearest neighbour that owns
  a date column at the right grain.

**2. sqlglot AST guardrails.** Every guarantee is made on the parsed tree, never
on a regex over SQL text, because string matching on SQL is defeated by comments,
string literals and whitespace. The guard is fail-closed:

- exactly one statement (a trailing `;` or comment is not a second one)
- root must be a `SELECT`, optionally with CTEs or a set operation
- no write, DDL, `PRAGMA`, `ATTACH` or transaction node anywhere in the tree
- no filesystem or extension-loading function
- every table and every qualified column must exist in the live catalog
- a row `LIMIT` is present, injected when missing and clamped when too large

Behind it, a second independent layer: the connection is opened read-only, a
SQLite **authorizer** denies every non-read action code at the driver level, and
a progress handler aborts anything over its wall-clock budget. A `DELETE` that
somehow got past the guard still fails with `authorizer_denied`.

**3. Verified-facts checker.** A number in a memo is grounded when it appears in
a result cell, or is one of the aggregates the checker recomputes itself: the row
count, a column total, one row's share of a column total, or a literal in the
executed query. Anything else is unverified, stripped from the memo, and recorded
on the trace. The bench plants a fabricated figure in three memos specifically to
confirm the checker catches it.

### The self-correction loop

A loop that can retry forever will eventually stumble onto something that runs,
which is not the same as being right. So the budget is explicit (`max_attempts`,
default 3), every attempt is logged with its SQL, guard verdict, error and
critique, and the bench scores how many attempts recovery cost.

The critic distinguishes repairable defects from dead ends. The most interesting
case is the one that is not an error at all:

```python
# vantage/agents/nodes.py
def _missing_group_by(sql: str) -> bool:
    """An aggregate projected beside a bare column with no GROUP BY.

    SQLite accepts this and quietly returns one arbitrary row, which is the
    nastiest failure in this pipeline: the query runs, the guard is happy, the
    shape is right and the number is wrong. Only the AST catches it.
    """
```

---

## vantage-bench

60 cases, four tiers, run through the real graph with nothing stubbed.

| Tier | Cases | Measures |
| --- | ---: | --- |
| 1 semantic execution accuracy | 24 | Does the answer match an independently hand-written gold query? Compared as row sets with floats rounded to 2dp, so aliasing and row order do not count against a correct answer. |
| 2 self-correction recovery | 12 | Seven distinct first-attempt faults are injected. A case passes only if the fault was **diagnosed** *and* the repaired answer matches gold. |
| 3 refusal correctness | 12 | Eight questions that must be refused, with the right category, and **four traps that must not be**. Scored as precision / recall / F1, because refusal rate alone is gameable by refusing everything. |
| 4 memo faithfulness | 12 | Every figure traced to the result set. Three cases plant a fabricated number; passing means the checker caught it and it never reached the memo. |

Also reported across all cases: schema linker recall, guardrail block rate, mean
attempts, and p50/p95 latency.

### Results

```
vantage-bench  model=mock  warehouse=258,000 rows  4.4s
==============================================================================
Tier 1  semantic execution accuracy    24/24 #################### 100.0%
Tier 2  self-correction recovery       12/12 #################### 100.0%
Tier 3  refusal correctness            12/12 #################### 100.0%
Tier 4  memo faithfulness              12/12 #################### 100.0%
------------------------------------------------------------------------------
overall                                60/60 #################### 100.0%

  refusal precision/recall/F1   1.00 / 1.00 / 1.00   (category accuracy 100%)
  self-correction recovery      100.0% over 12 injected faults, 2.0 attempts on average
  memo faithfulness (mean)      99.0% across 52 memos
  schema linker recall (mean)   100.0% (52/52 cases at 100%)
  latency p50 / p95             14ms / 124ms
```

### The controls that make those numbers mean something

A benchmark that only ever runs the system it was written for cannot tell you
whether it measures the system or the harness. Two controls bracket it, and both
run in CI:

```bash
python -m bench.runner --model null      # must score 0/60
python -m bench.runner --model oracle    # bounds the answer-correctness tiers
```

| Run | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `null` (valid SQL, no understanding, invented prose) | 0/24 | 0/12 | 0/12 | 0/12 | **0/60** |
| `oracle` (replays the hand-written gold query) | 24/24 | 12/12 | 4/12 | 12/12 | 52/60 |
| `mock` (the deterministic baseline) | 24/24 | 12/12 | 12/12 | 12/12 | **60/60** |

The oracle scoring 4/12 on refusal is the expected shape, not a defect: it has no
gold SQL for a question that should be refused, so it answers where it should
decline. CI fails the build if the null control ever passes a single case,
because that would mean a tier had stopped measuring anything.

### On the deterministic baseline

`--model mock` is a rule-based reader that produces the same plan JSON a hosted
model is asked for, then compiles it with the shared plan-to-SQL compiler. It
exists so the benchmark runs in CI with no API key, no network and no sampling
variance, through the same graph, guardrails and scorer as a hosted model.

It is a control, not a claim about difficulty. It saturates tiers 1 and 2 because
those cases sit inside the question grammar it implements, which is exactly what
makes it useful as a **regression gate**: a change that breaks the linker, the
guard or the critic turns the bench red immediately and offline. The interesting
number for a hosted model is how close it gets to the oracle on tier 1 and how it
scores on tiers 3 and 4, where the baseline's advantage does not apply.

Run a hosted model against the same suite:

```bash
export GEMINI_API_KEY=...
python -m bench.runner --model gemini --out bench/results-gemini
```

---

## Quick start

```bash
git clone <this repo> && cd vantage
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
vantage build                                    # 258,000 rows, ~1 second
vantage ask "Total revenue by product category in 2024"
```

No API key is needed. Everything above runs on the deterministic baseline.

### With a hosted or local model

```bash
cp .env.example .env        # then set VANTAGE_MODEL and the matching key
```

| `VANTAGE_MODEL` | Needs | Default model |
| --- | --- | --- |
| `mock` | nothing | deterministic baseline |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `ollama` | a local server | `llama3.1` |

Naming a provider without its key is a hard error rather than a silent downgrade:
a bench row that says `gemini` must have been produced by Gemini.

### Docker

```bash
docker compose up api                       # http://localhost:8000/docs
docker compose --profile bench run bench    # score the suite, write the report
```

The warehouse is generated at image build time, so the container runs offline and
its data is byte-identical to the one the benchmark was scored on.

---

## Interfaces

### CLI

```bash
vantage build                                  # generate the warehouse
vantage ask "revenue by category in 2024"
vantage ask "..." --trace                      # plan, linked tables, every attempt
vantage sql "SELECT ..." --check-only          # guard a query without running it
vantage schema orders                          # DDL for one table
vantage bench --model mock --fail-under 0.95
vantage serve                                  # FastAPI on :8000
vantage trace <trace_id>                       # replay a logged run
```

### HTTP

| Endpoint | Purpose |
| --- | --- |
| `POST /ask` | Answer a question. Returns the memo, the SQL, the rows **and the full trace**. |
| `POST /sql/validate` | Run the AST guard on a query without executing it. |
| `GET /schema` | Tables, grain, row counts, columns, foreign keys. |
| `GET /link?question=` | What the schema linker retrieves, with scores. For debugging recall. |
| `GET /traces/{id}` | Replay a logged run. |
| `GET /health` | Model, warehouse size, attempt budget, available providers. |

`/ask` returns the whole trace on purpose. An analyst who cannot see the query
has to take the number on trust, which is the thing this system exists to avoid.

### MCP

```bash
python -m vantage.mcp_server              # stdio
python -m vantage.mcp_server --http       # streamable HTTP on :8765
```

Six tools: `vantage_ask`, `vantage_schema`, `vantage_validate_sql`,
`vantage_run_sql`, `vantage_link`, `vantage_bench_summary`. The split matters:
`vantage_ask` is the whole graph, while `vantage_run_sql` is the raw execution
path and **still** runs the AST guard, so an agent holding that tool cannot write
to the warehouse whatever it sends.

---

## The warehouse

A retail order-to-cash schema in SQLite, generated deterministically from a fixed
seed so every number in this README is reproducible.

| Table | Rows | Grain |
| --- | ---: | --- |
| `order_items` | 118,000 | one row per product line on an order (the revenue fact) |
| `orders` | 42,000 | one row per order |
| `payments` | 44,000 | one row per payment attempt |
| `shipments` | 34,000 | one row per shipment |
| `customers` | 12,000 | one row per customer |
| `returns` | 5,200 | one row per returned line item |
| `products` | 2,500 | one row per SKU |
| `suppliers` | 200 | one row per supplier |
| `stores` | 60 | one row per store |
| `categories` | 40 | one row per category |
| **total** | **258,000** | |

The schema has two deliberate traps that a naive text-to-SQL pipeline falls into:
the revenue fact carries no date (the time axis is on `orders`), and a return rate
needs a `LEFT JOIN` or the denominator silently collapses to returned lines only.
Both are handled structurally rather than by special-casing, and both are in the
benchmark.

Column descriptions and business synonyms live in
`src/vantage/warehouse/glossary.yaml`; everything else is read from the database
by introspection, so the catalog cannot drift from the schema.

---

## Development

```bash
make install     # venv + dependencies + editable install
make warehouse   # generate data/warehouse.db
make test        # 73 tests
make bench       # score the suite
make api         # uvicorn with reload
```

### Layout

```
src/vantage/
  agents/          graph.py, nodes.py, prompts.py, state.py   the five agents
  guardrails/      sql_guard.py                               sqlglot AST validation
  llm/             base.py, mock.py, providers.py, registry.py the model seam
  retrieval/       linker.py, tfidf.py                        hybrid schema retrieval
  verify/          facts.py                                   verified-facts checker
  warehouse/       generate.py, catalog.py, glossary.yaml     data and schema catalog
  plan.py          the planner / SQL-writer contract
  sql_compiler.py  plan -> SQL, joins derived from the FK graph
  executor.py      read-only execution with an authorizer
  api.py           mcp_server.py   cli.py   run_log.py
bench/
  cases.yaml       60 cases across four tiers
  controls.py      null-agent and oracle controls
  metrics.py       scoring, refusal P/R/F1, nearest-rank percentiles
  runner.py        report.py
tests/             73 tests
```

### CI

Three jobs on every push: the test suite on Python 3.11 and 3.12; the benchmark
against the deterministic baseline with a `--fail-under` gate **plus** the
null-agent control that must score zero; and a Docker build whose image is
smoke-tested by asking it a real question over HTTP.

---

## Design notes

**The critic is not a model.** It reads the guard report, the execution outcome
and the parse tree. Those are mechanical facts, and a rule is both cheaper and
more reliable than a model at establishing them. A model is used for the memo,
where judgement actually helps.

**Joins are never guessed.** The compiler walks the catalog's foreign-key graph,
so it physically cannot emit a join predicate that does not exist in the schema.
That is what makes the deterministic baseline a fair control: both it and a
hosted model are held to the same plan, and only one of them can invent a
relationship.

**Nothing is overwritten.** A finished run carries every attempt: the SQL tried,
what the guard said, what the database said, what the critic decided next. That
record is what the bench scores, what `/ask` returns and what `vantage trace`
replays, and it is the only way to tell a model that got it right first time from
one that needed two repairs to reach the same answer.

**Refusals carry a category.** `out_of_scope`, `write_intent`, `pii_request`,
`unsupported_analysis`, `ambiguous`. The bench scores the category, not just the
refusal, because "I cannot forecast" and "that column does not exist" are
different answers and only one of them means the user should rephrase.

---

## License

MIT.
