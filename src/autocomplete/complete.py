"""Try GhostLine: complete the current line of any Python code.

Usage:
    uv run python -m autocomplete.complete "for i in range("
    uv run python -m autocomplete.complete --file my_script.py      # completes the end of the file
"""

import argparse
import time
from pathlib import Path

from autocomplete.data.tokens import detokenize, normalize, raw_tokens
from autocomplete.eval.decode import complete_fast
from autocomplete.models.kenlm_model import KenLM
from autocomplete.vocab import Vocab

ORDER = 6  # chosen on val: best perplexity and line metrics
BEAM = 4  # beam 8 gained < 1 point for clearly more work
THRESHOLD = 0.1  # chosen on val: precision 20% -> 41% for a small loss in characters saved


def context_tokens(code: str, vocab: Vocab) -> list[str]:
    """Tokens of the code before the cursor (the end of `code`)."""
    toks = normalize(raw_tokens(code, partial=True) or [], set(vocab.tokens))
    if not code.endswith("\n"):
        # The tokenizer closes the last line and blocks at end of input; the
        # user is still typing that line, so drop those closing markers.
        while toks and toks[-1] in ("<EOL>", "<DEDENT>"):
            toks.pop()
    return toks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code", nargs="?", help="code before the cursor")
    ap.add_argument("--file", type=Path)
    args = ap.parse_args()
    code = args.file.read_text() if args.file else (args.code or "")

    vocab = Vocab.load()
    model = KenLM(vocab, ORDER)
    context = vocab.encode(context_tokens(code, vocab))
    complete_fast(model, context, vocab, beam_size=BEAM)  # warm-up: loads the top-k tables once
    t = time.perf_counter()
    comp = complete_fast(model, context, vocab, beam_size=BEAM)
    ms = (time.perf_counter() - t) * 1000

    suggestion = detokenize([t for t in vocab.decode(comp.ids) if t != "<EOL>"])
    if not comp.ids:
        verdict = "silent: a string, number or unknown name comes next"
    elif comp.confidence >= THRESHOLD:
        verdict = "shown as ghost text"
    else:
        verdict = "silent: not confident enough"
    print(f"typed:      {code.splitlines()[-1] if code.strip() else ''}")
    print(f"best guess: {suggestion or '-'}")
    print(f"confidence: {comp.confidence:.2f} (threshold {THRESHOLD}) -> {verdict}   [{ms:.1f} ms]")


if __name__ == "__main__":
    main()
