import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "setup_m1_assets", Path(__file__).resolve().parents[1] / "scripts" / "setup_m1_assets.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_verify_accepts_expected_hashes():
    files = {k: {"sha256": v} for k, v in setup.EXPECTED_SHA256.items()}
    assert setup.verify(files) == []


def test_verify_reports_mismatch_and_missing():
    files = {k: {"sha256": v} for k, v in setup.EXPECTED_SHA256.items()}
    files["SlowFast.pyth"] = {"sha256": "0" * 64}
    del files["epic_slowfast_noun_reproduce/epoch_020.pth.tar"]
    bad = setup.verify(files)
    assert any(b.startswith("SlowFast.pyth: got") for b in bad)
    assert "epic_slowfast_noun_reproduce/epoch_020.pth.tar: missing" in bad


def test_vendored_source_is_not_checked_out_again(tmp_path, monkeypatch):
    """D-d: the source is committed without .git; `git -C` there would hit the ACTIVA repo."""
    (tmp_path / "actionformer_release").mkdir()
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: calls.append(a))
    assert setup.git_checkout(tmp_path / "actionformer_release", "url", "61ea7eb") == "61ea7eb"
    assert calls == []


def test_vendored_copies_keep_their_licences():
    root = Path(__file__).resolve().parents[1] / "weights" / "m1"
    for name in ("actionformer_release", "epic-kitchens-slowfast"):
        assert (root / name / "LICENSE").is_file(), name
