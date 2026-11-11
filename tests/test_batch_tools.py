"""scripts/run_manifest.py and scripts/validate_outputs.py on synthetic videos."""
import csv
import json
import sys
import types

import numpy as np
import pytest

import fallbacks
import run_manifest
import validate_outputs
import video_io
import vocab
from contracts import CategoryResult, ClipFeats, Importance


@pytest.fixture(autouse=True)
def stub_modules(monkeypatch):
    def clip(info):
        times = np.arange(0.0, info.duration, 0.5)
        feats = np.zeros((len(times), 4), np.float32)
        feats[:, 0] = 1.0
        return ClipFeats(times, feats, "test")

    stubs = {
        "clip_features": {"extract_clip_features": clip},
        "actions": {"detect_actions": lambda info: []},
        "importance": {"predict_importance": lambda info, c: Importance(c.times.copy(), np.full(len(c.times), 0.5))},
        "category": {"classify_video": lambda info, c, i: CategoryResult("dev-category-a", {"dev-category-a": 0.7})},
        "optimizer": {"select_highlights": lambda info, a, i, c: fallbacks.uniform_highlights(info)},
    }
    for name, functions in stubs.items():
        module = types.ModuleType(name)
        for key, fn in functions.items():
            setattr(module, key, fn)
        monkeypatch.setitem(sys.modules, name, module)


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "duration_sec", "dataset"])
        w.writerows(rows)
    return path


@pytest.fixture
def setup(media, dev_vocab, tmp_path):
    videos = tmp_path / "videos" / "nested"
    videos.mkdir(parents=True)
    for name in ("short", "black", "corrupt"):
        (videos / f"{name}.mp4").write_bytes(media[name].read_bytes())
    rows = [("short", f"{video_io.inspect_video(media['short']).duration:.2f}", "fpvsum"),
            ("black", f"{video_io.inspect_video(media['black']).duration:.2f}", "epic_kitchens_100"),
            ("corrupt", "60.00", "epic_kitchens_100"),
            ("not_downloaded", "100.00", "epic_kitchens_100")]
    manifest = write_manifest(tmp_path / "manifest.csv", rows)
    return tmp_path / "videos", manifest, tmp_path / "outputs", tmp_path / "report.json"


def statuses(report):
    return {e["video_id"]: e["status"] for e in report["entries"]}


def test_batch_run_reports_every_video_and_resumes(setup):
    videos, manifest, out, report_path = setup
    report = run_manifest.run_manifest(videos, out, manifest, report_path)
    assert statuses(report) == {"short": "ok", "black": "ok", "corrupt": "invalid_video",
                                "not_downloaded": "missing_video"}
    assert sorted(p.name for p in out.iterdir()) == ["black.json", "short.json"]
    assert json.loads(report_path.read_text())["summary"] == {"ok": 2, "invalid_video": 1, "missing_video": 1}

    again = run_manifest.run_manifest(videos, out, manifest, report_path)
    assert statuses(again)["short"] == "skipped_valid" and statuses(again)["black"] == "skipped_valid"
    forced = run_manifest.run_manifest(videos, out, manifest, report_path, dataset="fpvsum", force=True)
    assert statuses(forced) == {"short": "ok"}


def test_strict_batch_marks_fallbacks_as_failures(setup, monkeypatch):
    videos, manifest, out, report_path = setup
    monkeypatch.setenv("ACTIVA_STRICT", "1")
    monkeypatch.setitem(sys.modules, "actions", None)   # import fails
    report = run_manifest.run_manifest(videos, out, manifest, report_path, limit=1)
    assert statuses(report) == {"short": "strict_failure"} and not out.exists()


def test_duration_far_from_manifest_is_not_written(setup, tmp_path):
    videos, _, out, report_path = setup
    manifest = write_manifest(tmp_path / "m2.csv", [("short", "100.00", "fpvsum")])
    report = run_manifest.run_manifest(videos, out, manifest, report_path)
    assert statuses(report) == {"short": "invalid_output"}
    assert "duration_sec" in report["entries"][0]["error"] and not (out / "short.json").exists()


def test_validator_on_a_directory(setup, capsys):
    videos, manifest, out, report_path = setup
    run_manifest.run_manifest(videos, out, manifest, report_path)
    v = vocab.load()
    results = validate_outputs.validate_dir(out, v, validate_outputs.read_manifest(manifest))
    assert results["short.json"][0] == [] and results["black.json"][0] == []
    assert "2 manifest videos have no output" in results["<submission>"][0][0]
    assert validate_outputs.main(["--dir", str(out), "--manifest", str(manifest)]) == 1
    assert validate_outputs.main(["--file", str(out / "short.json"), "--manifest", str(manifest)]) == 0
    assert "local spec validator" in capsys.readouterr().out


def test_validator_catches_cross_file_problems(setup):
    videos, manifest, out, report_path = setup
    run_manifest.run_manifest(videos, out, manifest, report_path)
    data = json.loads((out / "black.json").read_text())
    data["team_id"] = "SOMEONE_ELSE"
    (out / "black.json").write_text(json.dumps(data))
    data["video_id"] = "short"
    (out / "renamed.json").write_text(json.dumps(data))
    results = validate_outputs.validate_dir(out, vocab.load(), validate_outputs.read_manifest(manifest))
    assert any("does not match file name" in e for e in results["renamed.json"][0])
    assert any("not in the manifest" in e for e in results["renamed.json"][0])
    assert any("team_id differs" in e for e in results["<submission>"][0])


def test_real_manifest_shape(kit_vocab):
    rows = validate_outputs.read_manifest(kit_vocab.parent / "manifest.csv")
    datasets = [ds for _, ds in rows.values()]
    assert len(rows) == 187 and datasets.count("epic_kitchens_100") == 138 and datasets.count("fpvsum") == 49
