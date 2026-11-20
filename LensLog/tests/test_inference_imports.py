"""run.sh never reaches development-only code (AGENTS.md: never read dev_cache/ from inference).

Walks the import graph statically from code/main.py: plain imports, lazy imports inside
functions, and module names passed as strings (main._call uses importlib). Over-counting is
fine; it only makes the check stricter.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
MODULES = {p.stem for p in CODE.glob("*.py")}

# Development tooling that must never run inside ./run.sh.
DEV_ONLY = {"fpvsum_data", "leakage_fingerprints", "perception_cache", "perception_cli",
            "perception_metrics", "tcn"}
# Paths that only exist on developer machines (caches, tuning and dev videos).
DEV_PATHS = ("dev_cache", "tuning_videos", "fpv_dev", "tmp/")


def _deps(module):
    tree = ast.parse((CODE / f"{module}.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in MODULES:
            found.add(node.value)
    return found & MODULES


def inference_modules():
    seen, todo = set(), ["main"]
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo += sorted(_deps(module) - seen)
    return seen


def test_dev_only_modules_exist():
    assert DEV_ONLY <= MODULES, "a dev-only module was renamed: update DEV_ONLY"


def test_every_stage_module_is_reached():
    assert {"clip_features", "importance", "category", "actions", "optimizer", "schema",
            "video_io", "vocab"} <= inference_modules()


def test_inference_never_imports_dev_only_code():
    assert inference_modules() & DEV_ONLY == set()


def test_inference_code_never_names_dev_paths():
    hits = [f"{m}.py: {p}" for m in sorted(inference_modules())
            for p in DEV_PATHS if p in (CODE / f"{m}.py").read_text(encoding="utf-8")]
    hits += [f"run.sh: {p}" for p in DEV_PATHS if p in (ROOT / "run.sh").read_text(encoding="utf-8")]
    assert hits == []
