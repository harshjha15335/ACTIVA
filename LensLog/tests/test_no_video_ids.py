"""No dataset video IDs in the inference code (AGENTS.md: never hardcode video IDs).

Covers every module under code/ plus run.sh. Comments count too: a judge reading the
source should not find manifest IDs in it.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "code").glob("*.py")) + [ROOT / "run.sh"]

EPIC_ID = re.compile(r"\bP\d{2}_\d{2,3}\b")  # P01_11, P02_109
FPVSUM_ID = re.compile(  # Bike_lhTFVa0h9Rk, BikePolo_5yf797Lh4Oc_2
    r"\b(?:Bike|Bikepolo|BikePolo|Boxing|Horse|Jumping|Longboarding|Motor|Parkour|Plane"
    r"|RockClimbing|Scuba|Skate|Ski|Surfing)_[A-Za-z0-9_-]{11}(?:_\d+)?\b")


def _hits(pattern):
    return [f"{p.relative_to(ROOT)}:{n}: {line.strip()}"
            for p in SOURCES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)]


def test_patterns_catch_real_ids():
    assert EPIC_ID.search('id = "P01_11"') and EPIC_ID.search("P02_109.json")
    assert FPVSUM_ID.search("Bike_lhTFVa0h9Rk") and FPVSUM_ID.search("BikePolo_5yf797Lh4Oc_2")
    assert not EPIC_ID.search("P100_1") and not FPVSUM_ID.search("Ski_short")


def test_no_epic_ids_in_inference_code():
    assert _hits(EPIC_ID) == []


def test_no_fpvsum_ids_in_inference_code():
    assert _hits(FPVSUM_ID) == []


def test_no_manifest_ids_when_the_kit_is_installed():
    manifest = ROOT / "kit" / "manifest.csv"
    if not manifest.exists():  # the kit stays local (D19); CI relies on the two regexes
        return
    with open(manifest, newline="", encoding="utf-8") as f:
        ids = [r["video_id"] for r in csv.DictReader(f)]
    assert len(ids) > 100
    found = [f"{p.relative_to(ROOT)}: {i}" for p in SOURCES
             for i in ids if re.search(rf"\b{re.escape(i)}\b", p.read_text(encoding="utf-8"))]
    assert found == []
