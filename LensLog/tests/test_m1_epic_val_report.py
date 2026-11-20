"""Synthetic check of scripts/m1_epic_val_report.py (no real annotations or outputs)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m1_epic_val_report", ROOT / "scripts" / "m1_epic_val_report.py")
rep_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep_mod)


def _fixture(tmp_path):
    cls = tmp_path / "cls"; cls.mkdir()
    (cls / "EPIC_100_verb_classes.csv").write_text("id,key\n0,take\n1,put\n", encoding="utf-8")
    (cls / "EPIC_100_noun_classes.csv").write_text("id,key\n0,cup\n1,plate\n", encoding="utf-8")
    ann = tmp_path / "val.csv"
    ann.write_text(
        "narration_id,video_id,start_timestamp,stop_timestamp,verb_class,noun_class\n"
        "a,VID_A,00:00:01.00,00:00:03.00,0,0\n"
        "b,VID_A,00:00:05.00,00:00:06.00,1,1\n"
        "c,VID_A,00:00:07.00,00:00:07.00,1,1\n"          # zero length: skipped
        "d,VID_B,00:00:01.00,00:00:02.00,0,1\n", encoding="utf-8")
    out = tmp_path / "outputs"; out.mkdir()
    (out / "VID_A.json").write_text(json.dumps({
        "video_id": "VID_A", "actions": [
            {"segment": [1.0, 3.0], "verb": "take", "noun": "cup", "score": 0.9},
            {"segment": [5.0, 6.0], "verb": "put", "noun": "plate", "score": 0.8}],
        "highlights": [{"segment": [0.0, 4.0]}]}), encoding="utf-8")
    (out / "OTHER.json").write_text(json.dumps({"video_id": "OTHER", "actions": []}), encoding="utf-8")
    return out, ann, cls


def test_dry_run_scores_nothing(tmp_path):
    out, ann, cls = _fixture(tmp_path)
    r = rep_mod.report(out, ann, cls, score=False)
    assert r["scored"] is False and "mAP" not in r
    assert r["annotations"]["gt_rows"] == 3 and r["annotations"]["skipped_zero_length_rows"] == 1
    assert r["outputs"]["matched_videos"] == 1
    assert r["outputs"]["annotated_videos_without_output"] == ["VID_B"]


def test_perfect_predictions_score_one(tmp_path):
    out, ann, cls = _fixture(tmp_path)
    r = rep_mod.report(out, ann, cls, score=True)
    assert r["mAP"]["pair"]["mAP"] == pytest.approx(1.0)
    assert r["mAP"]["verb"]["mAP"] == pytest.approx(1.0)
    cov = r["highlight_coverage_mean"]
    assert cov["any"] == pytest.approx(0.5) and cov["full"] == pytest.approx(0.5)


def test_refuses_annotations_inside_repo(tmp_path):
    with pytest.raises(SystemExit):
        rep_mod.main(["--annotations", str(ROOT / "EPIC_100_validation.csv")])
