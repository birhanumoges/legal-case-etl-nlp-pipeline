"""tests/conftest.py — pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to sys.path so all imports work from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
