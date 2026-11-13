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
