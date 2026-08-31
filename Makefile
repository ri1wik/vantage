PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install warehouse test bench api mcp docker clean

venv:
	python3.12 -m venv .venv && $(PIP) install -U pip

install: venv
	$(PIP) install -r requirements.txt && $(PIP) install -e .

warehouse:
	$(PY) -m vantage.warehouse.generate --out data/warehouse.db

test:
	$(PY) -m pytest -q

bench:
	$(PY) -m bench.runner --model mock --out bench/results

api:
	$(PY) -m uvicorn vantage.api:app --reload --port 8000

mcp:
	$(PY) -m vantage.mcp_server

docker:
	docker build -t vantage:latest .

clean:
	rm -rf .pytest_cache **/__pycache__ .coverage bench/results
