"""Test setup for the M1 branch.

`contracts.py` is owned by M4 and is not committed on this branch. If it is not
importable, tests use a test-only copy of the guide A5 `Action`/`VideoInfo`
dataclasses so the action engine can be tested on its own. Once the shared
foundation exists, the real module is used automatically.
"""
import sys
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

try:
    import contracts  # noqa: F401
except ImportError:
    @dataclass
    class VideoInfo:
        video_id: str
        path: str
        duration: float
        fps: float
        width: int
        height: int
        rotation: int = 0

    @dataclass
    class Action:
        verb: str
        noun: str
        start: float
        end: float
        score: float

    stub = types.ModuleType("contracts")
    stub.VideoInfo, stub.Action = VideoInfo, Action
    stub.__test_only__ = True
    sys.modules["contracts"] = stub
