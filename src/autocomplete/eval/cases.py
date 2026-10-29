"""Step 6a: build the frozen exam: random cursor positions in val / test / savio.

Each case = (file, cursor offset) + the target: the rest of that logical line,
up to and including <EOL>. Context is NOT copied: models read the file tokens
before the cursor and use as much as they can (KenLM ~4, the GRU hundreds).

    pick a random logical line with >= 3 real tokens (lines, not files: big files count more)
    cut it anywhere, keeping >= 1 real token to predict (the start of the line is allowed)
    no package may supply more than 10% of a split's cases

Labels stored per case, for the breakdowns:
    family_in_train      a related package (pydantic_core ~ pydantic) is in train  (option B)
    target_has_unk       contains <unk>: impossible to complete fully (we never suggest <unk>)
    target_has_literal   contains <STR>/<NUM>: the suggestion must stop before it

Usage:
    uv run python -m autocomplete.eval.cases
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from autocomplete import config
from autocomplete.vocab import read_files, read_meta

EVAL = config.DATA / "eval"
N_CASES = {"val": 2000, "test": 2000, "savio": 500}
PACKAGE_SHARE_CAP = 0.10
MIN_LINE_TOKENS = 3
STRUCTURE = {"<INDENT>", "<DEDENT>"}
PLACEHOLDERS = {"<unk>", "<STR>", "<NUM>"}


def family(package: str) -> str:
    """'pydantic_core' -> 'pydantic', 'savio:chatbot' -> 'chatbot', '_pytest' -> 'pytest'."""
    name = package.removeprefix("savio:").lstrip("_")
    return re.split(r"[_\-.]", name)[0].lower()


def logical_lines(tokens: list[str]):
    """Yield (start, end) of each logical line: tokens[start:end] ends with <EOL>.

    Leading <INDENT>/<DEDENT> belong to the context (block structure), not the line.
    """
    line_start = 0
    for i, tok in enumerate(tokens):
        if tok == "<EOL>":
            s = line_start
            while s < i and tokens[s] in STRUCTURE:
                s += 1
            yield s, i + 1
            line_start = i + 1


def build(split: str, train_families: set[str], rng: random.Random) -> list[dict]:
    files, meta = read_files(split), read_meta(split)
    lines = [
        (f, s, e)
        for f, toks in enumerate(files)
        for s, e in logical_lines(toks)
        if e - 1 - s >= MIN_LINE_TOKENS  # real tokens, <EOL> excluded
    ]
    rng.shuffle(lines)
    n = min(N_CASES[split], len(lines))
    cap = max(1, int(PACKAGE_SHARE_CAP * n))
    per_package: Counter = Counter()
    cases = []
    for f, s, e in lines:
        pkg = meta[f]["package"]
        if per_package[pkg] >= cap:
            continue
        per_package[pkg] += 1
        cut = rng.randrange(s, e - 1)  # leaves at least one real token before <EOL>
        target = files[f][cut:e]
        cases.append({
            "id": f"{split}-{len(cases):04d}",
            "file": f,
            "cursor": cut,
            "line_start": s,
            "target": target,
            "package": pkg,
            "family_in_train": family(pkg) in train_families,
            "target_has_unk": "<unk>" in target,
            "target_has_literal": bool({"<STR>", "<NUM>"} & set(target)),
        })
        if len(cases) == n:
            break
    return cases


def describe(split: str, cases: list[dict]) -> None:
    """The ceiling our own rules allow, before any model exists."""
    n = len(cases)
    full = sum(not (PLACEHOLDERS & set(c["target"])) for c in cases)
    abstain = sum(c["target"][0] in PLACEHOLDERS for c in cases)
    at_start = sum(c["cursor"] == c["line_start"] for c in cases)
    lens = sorted(len(c["target"]) for c in cases)
    print(
        f"{split:5s} {n:5,} cases  {len({c['package'] for c in cases}):3} packages  "
        f"full line possible {full / n:5.1%}  must stay silent {abstain / n:5.1%}  "
        f"cursor at line start {at_start / n:5.1%}  target len median {lens[n // 2]}  "
        f"related pkg in train {sum(c['family_in_train'] for c in cases) / n:5.1%}"
    )


def main() -> None:
    train_packages = json.loads((config.SPLITS / "packages.json").read_text())["train"]
    train_families = {family(p) for p in train_packages}
    EVAL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.SEED)
    for split in N_CASES:
        cases = build(split, train_families, rng)
        with (EVAL / f"{split}.jsonl").open("w") as f:
            f.writelines(json.dumps(c) + "\n" for c in cases)
        describe(split, cases)


def load_cases(split: str) -> list[dict]:
    return [json.loads(line) for line in Path(EVAL / f"{split}.jsonl").open()]


if __name__ == "__main__":
    main()
