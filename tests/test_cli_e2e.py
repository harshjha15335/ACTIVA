"""Runs ./run.sh in a subprocess, exactly as the evaluator would."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import schema
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


def run_sh(*args, vocab_path):
    env = {**os.environ, "ACTIVA_VOCAB": str(vocab_path), "ACTIVA_PYTHON": Path(sys.executable).as_posix()}
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
    assert data["video_id"] == name
    duration = {"short": 120, "long": 600, "black": 130, "vfr": 20, "rotated": 10}[name]
    schema.validate_output(data, duration, vocab.load(dev_vocab))
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
    proc = run_sh(media["short"], out, vocab_path=tmp_path / "no_vocab.json")
    assert proc.returncode == 5 and not out.exists()
    assert "ACTIVA_VOCAB" in proc.stderr


def test_run_sh_has_lf_line_endings():
    assert b"\r" not in RUN_SH.read_bytes()
