# ScandiQA

Scandinavian question-answering models and datasets.

Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/scandi_qa/index.html)
[![License](https://img.shields.io/github/license/alexadalab/dr-hatespeech)](https://github.com/alexandrainst/scandi-qa/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/scandi-qa)](https://github.com/alexandrainst/scandi-qa/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/scandi-qa/tree/main/tests)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```
make docs
```

To view the documentation, run:

```
make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```bash
.
├── config
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
├── data
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module
│   └── demo.py                     # demo module
└── tests                           # store tests
    └── __init__.py                 # make tests a Python module
```
