"""vocab.load() without the kit (D20): rebuilt from the EPIC-100 class CSVs + FPVSum categories."""
import json

import pytest

import config
import vocab
from vocab import VocabError

ROOT = config.ROOT
REAL_ASSETS = ROOT / "weights" / "m1"


def write_csvs(folder, verbs, nouns):
    folder.mkdir(parents=True, exist_ok=True)
    for kind, keys in (("verb", verbs), ("noun", nouns)):
        rows = ["id,key,instances,category"] + [f'{i},{k},"[\'{k}\']",x' for i, k in enumerate(keys)]
        (folder / f"EPIC_100_{kind}_classes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def no_kit(tmp_path, monkeypatch):
    """No ACTIVA_VOCAB and no kit file at the default location."""
    monkeypatch.delenv("ACTIVA_VOCAB", raising=False)
    monkeypatch.setattr(config, "DEFAULT_VOCAB_PATH", tmp_path / "kit" / "supported_classes.json")
    vocab._rebuilt_cached.cache_clear()
    yield
    vocab._rebuilt_cached.cache_clear()


def test_without_kit_the_vocabulary_is_rebuilt(no_kit, tmp_path, monkeypatch, caplog):
    assets = write_csvs(tmp_path / "m1", ["take", "put", "open"], ["tap", "plate", "extract:vanilla"])
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(assets))
    v = vocab.load()
    assert v.verbs == ("take", "put", "open", "other")                  # file order, then "other"
    assert v.nouns == ("tap", "plate", "extract:vanilla", "other")
    assert v.categories[-1] == "other" and len(v.categories) == 16
    assert list(v.categories[:-1]) == sorted(v.categories[:-1]) and "cooking" in v.categories
    assert v.is_valid_pair("open", "tap") and not v.is_valid_pair("open", "spoon")
    assert "rebuilt" in caplog.text


def test_without_kit_or_class_files_the_error_says_what_to_do(no_kit, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(tmp_path / "nothing"))
    with pytest.raises(VocabError, match="setup_weights.py"):
        vocab.load()


def test_an_explicit_missing_path_is_still_an_error(no_kit, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(write_csvs(tmp_path / "m1", ["take"], ["tap"])))
    monkeypatch.setenv("ACTIVA_VOCAB", str(tmp_path / "typo.json"))
    with pytest.raises(VocabError, match="ACTIVA_VOCAB"):
        vocab.load()
    with pytest.raises(VocabError):
        vocab.load(tmp_path / "typo.json")


def test_a_present_kit_file_wins(no_kit, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(write_csvs(tmp_path / "m1", ["take"], ["tap"])))
    config.DEFAULT_VOCAB_PATH.parent.mkdir(parents=True)
    config.DEFAULT_VOCAB_PATH.write_text(json.dumps({"labels": ["a"], "verbs": ["v"], "nouns": ["n"]}))
    assert vocab.load().categories == ("a",)


def test_rebuilt_vocabulary_equals_the_kit_file_exactly(kit_vocab, no_kit, monkeypatch):
    """Local only: needs the kit and the real EPIC-100 class files from setup_weights.py."""
    if not (REAL_ASSETS / "EPIC_100_verb_classes.csv").is_file():
        pytest.skip("EPIC-100 class files not installed (scripts/setup_weights.py)")
    monkeypatch.setenv("ACTIVA_M1_ASSETS", str(REAL_ASSETS))
    kit = json.loads(kit_vocab.read_text(encoding="utf-8"))
    rebuilt = vocab.load()
    assert list(rebuilt.categories) == kit["labels"]
    assert list(rebuilt.verbs) == kit["verbs"]
    assert list(rebuilt.nouns) == kit["nouns"]
