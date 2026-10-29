"""Steps 1-2: find every Python file we can learn from, then drop exact copies.

Sources, each file tagged with the PACKAGE it belongs to (step 4 splits by it):
    stdlib   .../python3.12/json/decoder.py             -> package "json"
    venv     .../site-packages/torch/nn/linear.py       -> package "torch"
    savio    ~/Desktop/train/chatbot/app.py             -> package "savio:chatbot"

Output: data/processed/files.jsonl (metadata only; step 3 reads the code).

Usage:
    uv run python -m autocomplete.data.collect
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from autocomplete import config

SKIP_DIRS = {"__pycache__", "node_modules", ".git", "site-packages", ".venv", "venv", "build", "dist"}


def stdlib_files() -> list[dict]:
    out = []
    for p in config.STDLIB.rglob("*.py"):
        rel = p.relative_to(config.STDLIB)
        if rel.parts[0] == "site-packages":
            continue
        package = rel.parts[0].removesuffix(".py")  # "json/decoder.py" -> json, "os.py" -> os
        out.append({"path": str(p), "source": "stdlib", "package": package})
    return out


def venv_files() -> list[dict]:
    out = []
    for site in sorted(config.SOURCE_ROOT.glob("*/.venv/lib/python*/site-packages")):
        if site.parts[len(config.SOURCE_ROOT.parts)] in config.EXCLUDE_PROJECTS:
            continue
        for p in site.rglob("*.py"):
            rel = p.relative_to(site)
            package = rel.parts[0].removesuffix(".py")  # "torch/nn/x.py" -> torch, "six.py" -> six
            out.append({"path": str(p), "source": "venv", "package": package})
    return out


def savio_files() -> list[dict]:
    """Your own code: every project folder, skipping venvs and build output."""
    out = []
    for project in sorted(p for p in config.SOURCE_ROOT.iterdir() if p.is_dir()):
        if project.name in config.EXCLUDE_PROJECTS or project.name.startswith("."):
            continue
        for p in project.rglob("*.py"):
            rel = p.relative_to(project)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            out.append({"path": str(p), "source": "savio", "package": f"savio:{project.name}"})
    return out


def main() -> None:
    found = stdlib_files() + venv_files() + savio_files()
    print(f"found {len(found):,} .py files  {dict(Counter(f['source'] for f in found))}")

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    config.FILES.parent.mkdir(parents=True, exist_ok=True)
    with config.FILES.open("w") as out:
        # Order matters for ties: a file that is both in stdlib and a venv keeps the stdlib tag.
        for f in found:
            try:
                data = Path(f["path"]).read_bytes()
            except OSError:
                stats["unreadable"] += 1
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen:
                stats["duplicate"] += 1
                continue
            seen.add(digest)
            stats[f"kept {f['source']}"] += 1
            out.write(json.dumps({**f, "sha1": digest, "bytes": len(data)}) + "\n")

    for k, v in sorted(stats.items()):
        print(f"  {k:12s} {v:>8,}")
    print(f"wrote {len(seen):,} unique files to {config.FILES}")


if __name__ == "__main__":
    main()
