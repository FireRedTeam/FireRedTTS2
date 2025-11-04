"""Utilities package for fireredtts2.

This module re-exports the commonly used helpers from the internal
`spliter` module so imports like `from fireredtts2.utils.spliter import ...`
continue to work when the package is installed.
"""

from .spliter import (
    clean_text,
    split_text,
    process_text,
    process_text_list,
)

__all__ = [
    "clean_text",
    "split_text",
    "process_text",
    "process_text_list",
]
