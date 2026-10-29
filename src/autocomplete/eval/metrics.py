"""Step 6c: metric definitions. Small pure functions, so they are easy to test."""

PLACEHOLDERS = {"<unk>", "<STR>", "<NUM>"}


def suggestible(target: list[str]) -> list[str]:
    """The part of the target our rules allow us to suggest: everything before
    the first <unk>/<STR>/<NUM>. Empty means the right move is to stay silent."""
    for i, tok in enumerate(target):
        if tok in PLACEHOLDERS:
            return target[:i]
    return target


def edit_distance(a: list[str], b: list[str]) -> int:
    """Token-level Levenshtein distance (insert / delete / substitute = 1)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_similarity(a: list[str], b: list[str]) -> float:
    """1 = identical, 0 = nothing in common. Two empty lists are identical."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1 - edit_distance(a, b) / longest


def length_bucket(n: int) -> str:
    return "1-3" if n <= 3 else "4-7" if n <= 7 else "8+"
