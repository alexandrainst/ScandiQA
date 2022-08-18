"""
.. include:: ../../README.md
"""

import pkg_resources

from .builder import QADatasetBuilder

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("scandi_qa").version
