"""Official vocabulary loader. Owned by M4.

    verbs, nouns, categories, is_valid_pair = vocab.load()

load() returns a 4-tuple (a NamedTuple, so attribute access also works), which
is the shape M2's category_vocab.py adapter expects. `categories` holds the
kit's video labels.

Layout of the student kit's supported_classes.json (version prism2026-theme01-v1):

    {"version": str, "labels": [16 str], "verbs": [98 str], "nouns": [301 str], "notes": str}

The kit lists no valid verb-noun pairs, so every known verb with every known
noun is accepted. An optional "pairs" list of [verb, noun] (used only by test
fixtures) restricts that.
"""
from functools import lru_cache
import json
from pathlib import Path
from typing import Callable, NamedTuple

import config


class VocabError(RuntimeError):
    """The vocabulary file is missing or not in the expected layout."""


class Vocab(NamedTuple):
    verbs: tuple
    nouns: tuple
    categories: tuple
    is_valid_pair: Callable[[str, str], bool]


def load(path=None) -> Vocab:
    path = Path(path) if path is not None else config.vocab_path()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise VocabError(
            f"Vocabulary file not found: {path}. Put the kit's supported_classes.json "
            "there or set ACTIVA_VOCAB to its path."
        ) from None
    return _load_cached(str(resolved), resolved.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _load_cached(path, _mtime):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocabError(f"Cannot read vocabulary file {path}: {error}") from None
    return _parse(data, path)


def _labels(data, key, path):
    labels = data.get(key)
    if not isinstance(labels, list) or not labels:
        raise VocabError(f"{path}: '{key}' must be a non-empty list of strings.")
    if any(not isinstance(x, str) or not x.strip() for x in labels):
        raise VocabError(f"{path}: '{key}' must contain only non-empty strings.")
    if len(set(labels)) != len(labels):
        raise VocabError(f"{path}: '{key}' contains duplicates.")
    return tuple(labels)


def _parse(data, path):
    if not isinstance(data, dict):
        raise VocabError(f"{path}: expected a JSON object with labels/verbs/nouns.")
    categories = _labels(data, "labels", path)
    verbs = _labels(data, "verbs", path)
    nouns = _labels(data, "nouns", path)
    verb_set, noun_set = frozenset(verbs), frozenset(nouns)

    pairs = data.get("pairs")
    if pairs is None:
        def is_valid_pair(verb, noun):
            return verb in verb_set and noun in noun_set
    else:
        if not isinstance(pairs, list):
            raise VocabError(f"{path}: 'pairs' must be a list of [verb, noun].")
        pair_set = set()
        for item in pairs:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise VocabError(f"{path}: every pair must be [verb, noun], got {item!r}.")
            verb, noun = item
            if verb not in verb_set or noun not in noun_set:
                raise VocabError(f"{path}: pair {item!r} uses an unknown verb or noun.")
            pair_set.add((verb, noun))
        pair_set = frozenset(pair_set)

        def is_valid_pair(verb, noun):
            return (verb, noun) in pair_set

    return Vocab(verbs, nouns, categories, is_valid_pair)
