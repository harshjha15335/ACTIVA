"""Local-only source screening; does not download media or claim leakage safety."""
import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m2_fpvsum_candidate_dry_run.py"
spec = importlib.util.spec_from_file_location("m2_fpvsum_dry_run", SCRIPT)
dry_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry_run)


def _write_csv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_excludes_original_source_and_aliases(tmp_path):
    inventory, manifest = (tmp_path / n for n in ["videos.csv", "manifest.csv"])
    _write_csv(inventory, ["video_name", "youtube_id"], [
        {"video_name": "protected", "youtube_id": "same-source"},
        {"video_name": "protected_reupload", "youtube_id": "same-source"},
        {"video_name": "safeA", "youtube_id": "unique1"},
        {"video_name": "safeA_alias", "youtube_id": "unique1"},
        {"video_name": "safeB", "youtube_id": "unique2"},
    ])
    _write_csv(manifest, ["video_id", "dataset"], [
        {"video_id": "protected", "dataset": "fpvsum"},
        {"video_id": "elsewhere", "dataset": "epic_kitchens_100"},
    ])
    result = dry_run.evaluate(inventory, manifest, False)
    assert result["manifest_fpvsum_ids"] == 1
    assert result["distinct_source_id_candidates"] == 2
    assert result["shared_youtube_source_exclusions"] == 1
    assert result["metadata_online_checked"] is False
    assert result["metadata_online_confirmed"] is None
    assert result["human_source_disjointness_review"] == "NOT DONE"
    assert not result["training_or_tuning_permission_granted"]
    assert result["duplicate_candidate_source_groups"] == {"unique1": ["safeA", "safeA_alias"]}
    assert {r["youtube_id"] for r in result["candidates"]} == {"unique1", "unique2"}
    assert all(r["clearance"] == "UNRESOLVED" for r in result["candidates"])


def test_missing_manifest_id_fails_closed(tmp_path):
    inventory, manifest = (tmp_path / n for n in ["videos.csv", "manifest.csv"])
    _write_csv(inventory, ["video_name", "youtube_id"], [
        {"video_name": "other", "youtube_id": "source"},
    ])
    _write_csv(manifest, ["video_id", "dataset"], [
        {"video_id": "manifest-only", "dataset": "fpvsum"},
    ])
    with pytest.raises(ValueError, match="Missing"):
        dry_run.evaluate(inventory, manifest, False)
