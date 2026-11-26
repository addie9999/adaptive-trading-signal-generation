.PHONY: install install-dev test lint format train predict serve backtest walkforward docker-up clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,backtest]"

test:
	pytest -q

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

train:
	python main.py train --ticker $(TICKER) --env research

predict:
	python main.py predict --ticker $(TICKER)

walkforward:
	python main.py walkforward --ticker $(TICKER) --model $(MODEL)

backtest:
	python main.py backtest --ticker $(TICKER) --model $(MODEL)

serve:
	python main.py serve --reload

docker-up:
	docker compose -f docker/docker-compose.yml up --build

clean:
	rm -rf artifacts mlruns reports data/raw data/processed .pytest_cache
