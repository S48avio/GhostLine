"""Step 6d: evaluate any model on the frozen exam and print every metric.

Usage:
    uv run python -m autocomplete.eval.run --model bigram --split val
    uv run python -m autocomplete.eval.run --model unigram --split test --limit 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from autocomplete import config
from autocomplete.data.tokens import _join
from autocomplete.eval.cases import load_cases
from autocomplete.eval.decode import complete_fast as complete  # proven identical to the plain beam search (300/300)
from autocomplete.eval.metrics import edit_similarity, length_bucket, suggestible
from autocomplete.models.baselines import Bigram, Unigram
from autocomplete.models.kenlm_model import KenLM
from autocomplete.vocab import Vocab, read_files

MODELS = {
    "unigram": Unigram,
    "bigram": Bigram,
    **{f"kenlm-{n}": (lambda vocab, n=n: KenLM(vocab, n)) for n in (3, 4, 5, 6)},
}  # the GRU joins in phase 2
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
MAX_CONTEXT = 512  # cap for long-context models (the GRU)
PERPLEXITY_TOKENS = 200_000  # perplexity on a seeded sample of files of about this size
RESULTS = Path("results")


def chars(tokens: list[str]) -> int:
    """Characters a user would type for these tokens."""
    return len(_join([t for t in tokens if t != "<EOL>"]))


def score_case(model, vocab: Vocab, case: dict, file_ids: list[int], beam: int) -> dict:
    n_ctx = min(model.context_size, MAX_CONTEXT)
    context = file_ids[max(0, case["cursor"] - n_ctx) : case["cursor"]] if n_ctx else []

    t = time.perf_counter()
    comp = complete(model, context, vocab, beam_size=beam)
    latency = (time.perf_counter() - t) * 1000

    # Next-token ranks with the TRUE previous tokens ("teacher forcing").
    # <unk> is never suggested, so a <unk> target always counts as a miss.
    target_ids = vocab.encode(case["target"])
    ranks = []
    for j, tid in enumerate(target_ids):
        logp = model.next_logprobs(context + target_ids[:j]).copy()
        logp[vocab.unk] = -np.inf
        ranks.append(float("inf") if tid == vocab.unk else 1 + int((logp > logp[tid]).sum()))

    sugg, target = vocab.decode(comp.ids), case["target"]
    goal = suggestible(target)
    return {
        "id": case["id"],
        "suggestion": sugg,
        "confidence": comp.confidence,
        "full_exact": sugg == target,
        "sugg_exact": sugg == goal,
        "edit_sim": edit_similarity(sugg, goal),
        "should_be_silent": not goal,
        "ranks": ranks,
        "latency_ms": latency,
        "target_chars": chars(target),
        "sugg_chars": chars(sugg),
        "family_in_train": case["family_in_train"],
        "bucket": length_bucket(len(target)),
    }


def perplexity(model, vocab: Vocab, files: list[list[str]]) -> float:
    order = list(range(len(files)))
    random.Random(config.SEED).shuffle(order)
    total, n = 0.0, 0
    for f in order:
        lp = model.stream_logprobs(np.array(vocab.encode(files[f]), dtype=np.int64))
        total += float(lp.sum())
        n += len(lp)
        if n >= PERPLEXITY_TOKENS:
            break
    return float(np.exp(-total / n))


def summarize(rows: list[dict]) -> dict:
    ranks = [r for row in rows for r in row["ranks"]]
    out = {
        "cases": len(rows),
        "acc@1": float(np.mean([r <= 1 for r in ranks])),
        "acc@5": float(np.mean([r <= 5 for r in ranks])),
        "full_exact": float(np.mean([r["full_exact"] for r in rows])),
        "sugg_exact": float(np.mean([r["sugg_exact"] for r in rows])),
        "edit_sim": float(np.mean([r["edit_sim"] for r in rows])),
    }
    silent = [r for r in rows if r["should_be_silent"]]
    out["correctly_silent"] = float(np.mean([not r["suggestion"] for r in silent])) if silent else float("nan")
    return out


def threshold_table(rows: list[dict]) -> list[dict]:
    """Coverage / precision / chars saved if we only show suggestions with confidence >= t."""
    total_chars = sum(r["target_chars"] for r in rows)
    table = []
    for t in THRESHOLDS:
        shown = [r for r in rows if r["suggestion"] and r["confidence"] >= t]
        right = [r for r in shown if r["sugg_exact"]]
        table.append({
            "threshold": t,
            "coverage": len(shown) / len(rows),
            "precision": len(right) / len(shown) if shown else float("nan"),
            "chars_saved": sum(r["sugg_chars"] for r in right) / total_chars,
        })
    return table


def evaluate(model, split: str, vocab: Vocab, limit: int | None, beam: int) -> dict:
    cases = load_cases(split)[:limit]
    files = read_files(split)
    encoded: dict[int, list[int]] = {}
    rows = []
    for case in cases:
        if case["file"] not in encoded:
            encoded[case["file"]] = vocab.encode(files[case["file"]])
        rows.append(score_case(model, vocab, case, encoded[case["file"]], beam))

    lat = sorted(r["latency_ms"] for r in rows)
    result = {
        "model": model.name,
        "split": split,
        "beam": beam,
        "perplexity": perplexity(model, vocab, files),
        "overall": summarize(rows),
        "latency_p50_ms": lat[len(lat) // 2],
        "latency_p95_ms": lat[int(len(lat) * 0.95)],
        "thresholds": threshold_table(rows),
        "by_family": {str(k): summarize([r for r in rows if r["family_in_train"] == k]) for k in (False, True)},
        "by_length": {b: summarize([r for r in rows if r["bucket"] == b]) for b in ("1-3", "4-7", "8+")},
    }
    RESULTS.mkdir(exist_ok=True)
    stem = f"{model.name}-{split}" + ("" if beam == 4 else f"-beam{beam}")
    (RESULTS / f"{stem}.json").write_text(json.dumps(result, indent=1))
    with (RESULTS / f"{stem}.cases.jsonl").open("w") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != "ranks"}) + "\n" for r in rows)
    return result


def report(res: dict) -> None:
    o = res["overall"]
    print(f"\n== {res['model']} on {res['split']} ({o['cases']} cases, beam {res['beam']}) ==")
    print(f"perplexity {res['perplexity']:.1f}   acc@1 {o['acc@1']:.1%}   acc@5 {o['acc@5']:.1%}")
    print(f"full-line exact {o['full_exact']:.1%}   suggestible exact {o['sugg_exact']:.1%}   "
          f"edit sim {o['edit_sim']:.2f}   correctly silent {o['correctly_silent']:.1%}")
    print(f"latency p50 {res['latency_p50_ms']:.2f} ms   p95 {res['latency_p95_ms']:.2f} ms")
    print("  threshold  coverage  precision  chars saved")
    for row in res["thresholds"]:
        print(f"  {row['threshold']:9.2f}  {row['coverage']:8.1%}  {row['precision']:9.1%}  {row['chars_saved']:11.1%}")
    for title, groups in (("related package in train", res["by_family"]), ("target length", res["by_length"])):
        print(f"  by {title}: " + "   ".join(
            f"{k}: n={g['cases']} sugg-exact {g['sugg_exact']:.1%} acc@1 {g['acc@1']:.1%}" for k, g in groups.items() if g["cases"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--split", choices=["val", "test", "savio"], default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()
    vocab = Vocab.load()
    t = time.time()
    model = MODELS[args.model](vocab)
    print(f"built {args.model} in {time.time() - t:.0f}s")
    report(evaluate(model, args.split, vocab, args.limit, args.beam))


if __name__ == "__main__":
    main()
