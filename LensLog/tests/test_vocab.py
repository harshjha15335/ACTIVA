import json

import pytest

import vocab
from vocab import VocabError


def write(tmp_path, data):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(data))
    return path


BASE = {"verbs": ["take", "put"], "nouns": ["cup", "knife"], "labels": ["cooking", "walking"]}


def test_load_returns_four_tuple_m2_adapter_can_unpack(dev_vocab):
    loaded = vocab.load()
    assert isinstance(loaded, tuple) and len(loaded) == 4   # m2 category_vocab.py checks this
    verbs, nouns, categories, is_valid_pair = loaded
    assert categories == loaded.categories
    assert "dev-category-a" in categories and callable(is_valid_pair)


def test_env_variable_overrides_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVA_VOCAB", str(write(tmp_path, BASE)))
    assert vocab.load().categories == ("cooking", "walking")


def test_missing_file_explains_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVA_VOCAB", str(tmp_path / "nope.json"))
    with pytest.raises(VocabError, match="ACTIVA_VOCAB"):
        vocab.load()


def test_without_pairs_any_known_verb_noun_combination_is_valid(tmp_path):
    v = vocab.load(write(tmp_path, BASE))
    assert v.is_valid_pair("take", "knife")
    assert not v.is_valid_pair("take", "spoon")
    assert not v.is_valid_pair("cut", "cup")


def test_pairs_restrict_combinations(tmp_path):
    v = vocab.load(write(tmp_path, {**BASE, "pairs": [["take", "cup"]]}))
    assert v.is_valid_pair("take", "cup")
    assert not v.is_valid_pair("put", "cup")


@pytest.mark.parametrize("bad", [
    {**BASE, "verbs": []},
    {**BASE, "nouns": ["cup", "cup"]},
    {**BASE, "labels": ["ok", ""]},
    {**BASE, "labels": "cooking"},
    {"verbs": ["take"], "nouns": ["cup"]},
    {**BASE, "pairs": [["take", "spoon"]]},
    {**BASE, "pairs": [["take"]]},
    ["take", "put"],
])
def test_bad_layouts_are_rejected(tmp_path, bad):
    with pytest.raises(VocabError):
        vocab.load(write(tmp_path, bad))


def test_unreadable_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(VocabError):
        vocab.load(path)


def test_edited_file_is_reloaded(tmp_path):
    path = write(tmp_path, BASE)
    assert vocab.load(path).categories == ("cooking", "walking")
    import os
    path.write_text(json.dumps({**BASE, "labels": ["cleaning"]}))
    os.utime(path, ns=(1, 1))   # force a different mtime
    assert vocab.load(path).categories == ("cleaning",)


def test_layout_matches_kit_and_needs_labels(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"version": "x", "labels": ["a", "other"], "verbs": ["v"], "nouns": ["n"], "notes": ""}))
    loaded = vocab.load(path)
    assert loaded.categories == ("a", "other") and loaded.is_valid_pair("v", "n")
    path.write_text(json.dumps({"categories": ["a"], "verbs": ["v"], "nouns": ["n"]}))
    with pytest.raises(vocab.VocabError, match="labels"):
        vocab.load(path)


def test_real_kit_vocabulary(kit_vocab):
    v = vocab.load(kit_vocab)
    assert (len(v.categories), len(v.verbs), len(v.nouns)) == (16, 98, 301)
    assert {"cooking", "other", "ski"} <= set(v.categories) and "skiing" not in v.categories
    assert v.is_valid_pair("open", "fridge") and v.is_valid_pair("other", "other")
