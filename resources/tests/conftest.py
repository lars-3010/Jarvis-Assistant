"""Pytest configuration for resources/tests.

Ensures the repository root is on sys.path so tests can import
helpers via absolute package path like `resources.tests.helpers`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

