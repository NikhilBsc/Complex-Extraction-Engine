"""
ingestion/__init__.py
---------------------
Public surface of the ingestion package.

Only expose what downstream layers actually need:
  - read_pdf  → the main entry point
  - PageResult → the standardised per-page data contract
"""

from ingestion.reader import PageResult, read_pdf

__all__ = ["read_pdf", "PageResult"]
