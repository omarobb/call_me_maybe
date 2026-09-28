.PHONY: install env run clean debug lint 
path = $(shell pwd)

all: run

env:
	@read -p "Are you on 42 device [y|n] " d; \
	if [ "$$d" = "y" ]; then \
		cd /goinfre/$(USER) && \
		mkdir -p call && \
		cd call && \
		python3 -m venv .venv && \
		ln -s -f /goinfre/$(USER)/call/.venv $(path) && \
		export UV_LINK_MODE=copy; \
	else \
		mkdir -p call_me_maybe && \
		cd call_me_maybe && \
		python3 -m venv .venv; \
	fi

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync

run:
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf ./data/output
	rm -rf .mypy_cache

debug:
	uv run python -m pdb -m src

test:
	uv run python -m moulinette grade_student_answers --set private --student_answer_path ../data/output/function_calling_results.json

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
