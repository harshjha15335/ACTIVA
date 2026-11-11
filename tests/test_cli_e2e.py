"""Runs ./run.sh in a subprocess, exactly as the evaluator would."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import schema
import video_io
import vocab

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "run.sh"


def find_bash():
    if os.name == "nt":   # avoid WSL's System32\bash.exe; use Git Bash
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
        return None
    return shutil.which("bash")


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def run_sh(*args, vocab_path, **extra_env):
    env = {**os.environ, "ACTIVA_VOCAB": str(vocab_path), "ACTIVA_PYTHON": Path(sys.executable).as_posix(),
           **extra_env}
    posix = [Path(a).as_posix() if isinstance(a, Path) else a for a in args]
    return subprocess.run([BASH, RUN_SH.as_posix(), *posix], capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.parametrize("name", ["short", "long", "black", "vfr", "rotated"])
def test_good_videos_give_valid_json(media, dev_vocab, tmp_path, name):
    out = tmp_path / "nested" / f"{name}.json"
    proc = run_sh(media[name], out, vocab_path=dev_vocab)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", "logs must go to stderr"
    assert "stage" in proc.stderr
    data = json.loads(out.read_text())
    assert data["video_id"] == name and data["team_id"] == "TEST_TEAM"
    duration = video_io.inspect_video(media[name]).duration
    schema.validate_output(data, duration, vocab.load(dev_vocab), team_id="TEST_TEAM")
    assert [p.name for p in out.parent.iterdir()] == [out.name], "temp file left behind"


@pytest.mark.parametrize("name", ["corrupt", "empty", "audio_only", "not_video"])
def test_bad_videos_fail_without_json(media, dev_vocab, tmp_path, name):
    out = tmp_path / "out.json"
    proc = run_sh(media[name], out, vocab_path=dev_vocab)
    assert proc.returncode == 4, proc.stderr
    assert not out.exists()
    assert "invalid input video" in proc.stderr


def test_missing_input_exit_3(dev_vocab, tmp_path):
    proc = run_sh(tmp_path / "missing.mp4", tmp_path / "out.json", vocab_path=dev_vocab)
    assert proc.returncode == 3 and not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("args", [(), ("only_one.mp4",), ("a.mp4", "b.json", "c")])
def test_wrong_argument_count_exit_2(dev_vocab, args):
    proc = run_sh(*args, vocab_path=dev_vocab)
    assert proc.returncode == 2 and "Usage" in proc.stderr


def test_missing_vocabulary_exit_5(media, tmp_path):
    out = tmp_path / "out.json"
    proc = run_sh(media["short"], out, vocab_path=tmp_path / "no_vocab.json", ACTIVA_TEAM_ID="T")
    assert proc.returncode == 5 and not out.exists()
    assert "ACTIVA_VOCAB" in proc.stderr


def test_missing_team_id_exit_5(media, dev_vocab, tmp_path, monkeypatch):
    monkeypatch.delenv("ACTIVA_TEAM_ID")
    out = tmp_path / "out.json"
    proc = run_sh(media["short"], out, vocab_path=dev_vocab)
    if "TEAM_ID = None" not in (ROOT / "code" / "config.py").read_text():
        pytest.skip("config.TEAM_ID is set")
    assert proc.returncode == 5 and not out.exists()
    assert "team_id is not set" in proc.stderr


def test_strict_mode_exit_6_when_a_stage_fails(media, dev_vocab, tmp_path):
    """No CLIP weights at this path, so the clip stage fails; strict mode must not hide it."""
    out = tmp_path / "out.json"
    proc = run_sh(media["short"], out, vocab_path=dev_vocab, ACTIVA_STRICT="1",
                  ACTIVA_CLIP_WEIGHTS=(tmp_path / "missing.safetensors").as_posix())
    assert proc.returncode == 6, proc.stderr
    assert not out.exists() and "strict mode" in proc.stderr


def test_run_sh_has_lf_line_endings():
    assert b"\r" not in RUN_SH.read_bytes()
