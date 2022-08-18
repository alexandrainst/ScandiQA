# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: notebook docs

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Includes environment variables from the .env file
include .env

install-poetry:
	@echo "Installing poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
	@$(eval include ${HOME}/.poetry/env)

uninstall-poetry:
	@echo "Uninstalling poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - --uninstall

install:
	@echo "Installing..."
	@if [ "$(shell which poetry)" = "" ]; then \
		$(MAKE) install-poetry; \
	fi
	@if [ "$(shell which gpg)" = "" ]; then \
		echo "GPG not installed, so an error will occur. Install GPG on MacOS with "\
			 "`brew install gnupg` or on Ubuntu with `apt install gnupg` and run "\
			 "`make install` again."; \
	fi
	@$(MAKE) setup-poetry
	@$(MAKE) setup-git

setup-poetry:
	@poetry env use python3
	@poetry run python3 -m src.scripts.fix_dot_env_file
	@poetry install
	@poetry run pre-commit install

setup-git:
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}
	@if [ ${GPG_KEY_ID} = "" ]; then \
		echo "No GPG key ID specified. Skipping GPG signing."; \
		git config --local commit.gpgsign false; \
	else \
		echo "Signing with GPG key ID ${GPG_KEY_ID}..."; \
		echo 'If you get the "failed to sign the data" error when committing, try running `export GPG_TTY=$$(tty)`.'; \
		git config --local commit.gpgsign true; \
		git config --local user.signingkey ${GPG_KEY_ID}; \
	fi

docs:
	@poetry run pdoc --docformat google src/scandi_qa -o docs
	@echo "Saved documentation."

view-docs:
	@echo "Viewing API documentation..."
	@open docs/scandi_qa.html

bump-major:
	@poetry run python -m src.scripts.versioning --major
	@echo "Bumped major version."

bump-minor:
	@poetry run python -m src.scripts.versioning --minor
	@echo "Bumped minor version."

bump-patch:
	@poetry run python -m src.scripts.versioning --patch
	@echo "Bumped patch version."

publish:
	@$(eval include .env)
	@printf "Preparing to publish to PyPI. Have you ensured to change the package version with 'make bump-X' for 'X' being 'major', 'minor' or 'patch'? [y/n] : "; \
		read -r answer; \
		if [ "$${answer}" = "y" ]; then \
			if [ "$${PYPI_API_TOKEN}" = "" ]; then \
				echo "No PyPI API token specified in the '.env' file, so cannot publish."; \
			else \
				echo "Publishing to PyPI..."; \
				poetry publish --build --username "__token__" --password "$${PYPI_API_TOKEN}"; \
				echo "Published!"; \
			fi \
		else \
			echo "Publishing aborted."; \
		fi

test:
	@pytest && readme-cov

tree:
	@tree -a \
		-I .git \
		-I .mypy_cache . \
		-I .env \
		-I .venv \
		-I poetry.lock \
		-I .ipynb_checkpoints \
		-I dist \
		-I .gitkeep \
		-I docs \
		-I .pytest_cache \
		-I outputs \
		-I .DS_Store \
		-I .cache \
		-I *.parquet \
		-I *.csv \
		-I *.txt \
		-I checkpoint-* \
		-I .coverage*
