PY=.venv/bin/python
ENV?=.venv
EXP?=churn-exp

init:
	python3 -m venv $(ENV) && $(ENV)/bin/pip install -U pip && $(ENV)/bin/pip install -r requirements.txt

train:
	PYTHONPATH=. MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/train.py --config configs/config.yaml

evaluate:
	PYTHONPATH=. MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) src/evaluate.py --config configs/config.yaml

test:
	PYTHONPATH=. $(ENV)/bin/pytest -q

lint:
	$(ENV)/bin/ruff check .

ui:
	$(ENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
