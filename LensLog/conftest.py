"""Shared pytest fixtures for M4's integration tests.

Lives at the repo root (not tests/conftest.py) so it never collides with a
member's own tests/conftest.py. Fixtures are opt-in, so members' tests are
unaffected unless they ask for them.
"""
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

DEV_VOCAB = ROOT / "tests" / "fixtures" / "dev_vocab.json"

# CI has no model weights: let run.sh fall back stage by stage instead of exiting 5
# (main.missing_model_files). Tests of that check set the variable themselves.
os.environ.setdefault("ACTIVA_REQUIRE_MODELS", "0")


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """Synthetic videos: short (2 min), long (10 min), black, vfr, rotated, corrupt, ..."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    from make_fixtures import make_fixtures
    return make_fixtures(tmp_path_factory.mktemp("media"))


TEST_TEAM_ID = "TEST_TEAM"
KIT_VOCAB = ROOT / "kit" / "supported_classes.json"


@pytest.fixture
def dev_vocab(monkeypatch):
    """Point vocab.load() at the placeholder test vocabulary and set a test team_id."""
    monkeypatch.setenv("ACTIVA_VOCAB", str(DEV_VOCAB))
    monkeypatch.setenv("ACTIVA_TEAM_ID", TEST_TEAM_ID)
    monkeypatch.delenv("ACTIVA_STRICT", raising=False)
    return DEV_VOCAB


@pytest.fixture
def kit_vocab():
    """The real student-kit vocabulary, if it is installed in kit/ (never in CI)."""
    if not KIT_VOCAB.exists():
        pytest.skip("student kit not installed in kit/")
    return KIT_VOCAB


@pytest.fixture
def kit_file():
    """kit_file(name): another kit file (manifest, examples). Local-only, so the test skips when absent (CI)."""
    def get(name):
        path = KIT_VOCAB.parent / name
        if not path.exists():
            pytest.skip(f"kit/{name} is local-only (not committed)")
        return path
    return get
