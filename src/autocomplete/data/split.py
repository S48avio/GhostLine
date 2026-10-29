"""Step 4: split by PACKAGE into train / val / test (~80/10/10 of code size).

Why by package: if torch/nn/linear.py is in train and torch/nn/conv.py in
test, the model has already seen torch's names and style, so the test would
measure "torch memory", not autocomplete skill. Holding out whole packages
asks the honest question: how well does it complete a library it never saw?

Packages differ hugely in size, so a coin flip per package would give uneven
splits. Instead: shuffle packages (seeded), then give each one to the split
that is furthest below its target share. Size = characters of code, a proxy
until step 5 counts real tokens.

Your own code (savio:*) becomes a separate test set, never trained on.

Usage:
    uv run python -m autocomplete.data.split
"""

import json
import random
from collections import defaultdict

from autocomplete import config

TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}


def assign(sizes: dict[str, int]) -> dict[str, str]:
    """package -> split, each split close to its target share of the total size."""
    packages = sorted(sizes)
    random.Random(config.SEED).shuffle(packages)
    total = sum(sizes.values())
    filled = dict.fromkeys(TARGETS, 0)
    out = {}
    for pkg in packages:
        # The split with the biggest gap between target and current share gets it.
        split = max(TARGETS, key=lambda s: TARGETS[s] - filled[s] / total)
        out[pkg] = split
        filled[split] += sizes[pkg]
    return out


def main() -> None:
    records = [json.loads(line) for line in config.CLEAN.open()]
    sizes: dict[str, int] = defaultdict(int)
    for r in records:
        if not r["package"].startswith("savio:"):
            sizes[r["package"]] += len(r["code"])
    split_of = assign(sizes)

    parts: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        parts["savio" if r["package"].startswith("savio:") else split_of[r["package"]]].append(r)

    config.SPLITS.mkdir(parents=True, exist_ok=True)
    total = sum(sizes.values())
    for name in ("train", "val", "test", "savio"):
        rows = parts[name]
        with (config.SPLITS / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        chars = sum(len(r["code"]) for r in rows)
        pkgs = sorted({r["package"] for r in rows}, key=lambda p: -sum(len(r["code"]) for r in rows if r["package"] == p))
        share = f"{chars / total:5.1%}" if name != "savio" else "  -  "
        print(f"{name:5s} {len(rows):6,} files  {len(pkgs):4,} packages  {chars / 1e6:6.1f} MB  {share}  biggest: {', '.join(pkgs[:6])}")

    # Record which package went where, so the split is inspectable and stable.
    (config.SPLITS / "packages.json").write_text(
        json.dumps({s: sorted(p for p, v in split_of.items() if v == s) for s in TARGETS}, indent=1)
    )


if __name__ == "__main__":
    main()
