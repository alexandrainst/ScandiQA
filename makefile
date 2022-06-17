.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

install-poetry:
	@echo "Installing poetry..."
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

activate:
	@echo "Activating virtual environment..."
	@poetry shell
	@source `poetry env info --path`/bin/activate

install:
	@echo "Installing..."
	@git init
	@git config commit.gpgsign true
	@git config user.signingkey "D3163F7C12AE2EFBCE98058FE0E6DFBD1D28BC10"
	@git config user.email "dan.nielsen@alexandra.dk"
	@git config user.name "Dan Saattrup Nielsen"
	@poetry install
	@poetry run pre-commit install

remove-env:
	@poetry env remove python3
	@echo "Removed virtual environment."

view-docs:
	@echo "Viewing API documentation..."
	@poetry run pdoc src/scandi_qa

docs:
	@poetry run pdoc src/scandi_qa -o docs
	@echo "Saved documentation."

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@echo "Cleaned repository."

