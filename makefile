.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:
export ENV_DIR="$( poetry env list --full-path | grep Activated | cut -d' ' -f1 )"

activate:
	@echo "Activating virtual environment"
	@poetry shell
	@source "$(ENV_DIR)/bin/activate"

install:
	@echo "Installing..."
	@git init
	@poetry install
	@poetry run pre-commit install

delete_env:
	@poetry env remove python3

docs_view:
	@echo View API documentation...
	@pdoc src/scandi_qa

docs:
	@echo Save documentation to docs...
	@pdoc src/scandi_qa -o docs

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache

build_dataset:
	@echo "Building dataset..."
	@python -m src.scandi_qa.builder
