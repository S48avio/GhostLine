"""Paths and constants shared by every step. Run commands from the project root."""

import sysconfig
from pathlib import Path

# Where the Python code we learn from lives.
SOURCE_ROOT = Path.home() / "Desktop" / "train"  # your projects and their .venv folders
STDLIB = Path(sysconfig.get_paths()["stdlib"])  # Python's own standard library
EXCLUDE_PROJECTS = {"autocomplete"}  # this project: its code and venv change as we work

DATA = Path("data")
FILES = DATA / "processed" / "files.jsonl"  # step 1-2: collected + deduped (metadata only)
CLEAN = DATA / "processed" / "clean.jsonl"  # step 3: filtered, WITH code text (frozen copy)
SPLITS = DATA / "splits"  # step 4: train / val / test / savio .jsonl

SEED = 42
