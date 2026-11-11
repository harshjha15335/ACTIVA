"""One-time setup of the M1 action-engine assets (needs internet; run.sh does not).

    python scripts/setup_m1_assets.py [--dest weights/m1]

Creates the EngineAssets layout from code/action_config.py:
  epic-kitchens-slowfast/   git @ bf50519   (Apache-2.0)  SlowFast model code
  SlowFast.pyth             EPIC-100 SlowFast 8x8 R50 checkpoint (epic-kitchens-slowfast README)
  actionformer_release/     git @ 61ea7eb   (MIT)         ActionFormer model code
  epic_slowfast_{verb,noun}_reproduce/   ActionFormer EPIC checkpoints (ActionFormer README)
  EPIC_100_{verb,noun}_classes.csv       class id -> key (epic-kitchens-100-annotations)
and writes ASSETS.json with the source URL, size and SHA-256 of every file it fetched.
Existing files are kept (re-run is cheap). Nothing here is used at inference time
except the files themselves.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GIT = {
    "epic-kitchens-slowfast": ("https://github.com/epic-kitchens/epic-kitchens-slowfast.git", "bf50519"),
    "actionformer_release": ("https://github.com/happyharrycn/actionformer_release.git", "61ea7eb"),
}
URLS = {
    "SlowFast.pyth": "https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=1",
    "EPIC_100_verb_classes.csv": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_verb_classes.csv",
    "EPIC_100_noun_classes.csv": "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes.csv",
}
GDRIVE = {  # ActionFormer README "Evaluating pretrained models"
    "epic_slowfast_verb_reproduce": "1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4",
    "epic_slowfast_noun_reproduce": "1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_checkout(dest, url, commit):
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def fetch(url, dest):
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)


def fetch_gdrive_zip(file_id, out_dir, work):
    if any(out_dir.glob("*.pth.tar")):
        return
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown  (needed once to fetch the ActionFormer checkpoints)")
    work.mkdir(parents=True, exist_ok=True)
    zpath = Path(gdown.download(id=file_id, output=str(work) + "/", quiet=False))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(work / zpath.stem)
    ckpts = sorted((work / zpath.stem).rglob("*.pth.tar"))
    if not ckpts:
        sys.exit(f"no .pth.tar inside {zpath}")
    shutil.copytree(ckpts[0].parent, out_dir, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "weights" / "m1"))
    dest = Path(ap.parse_args().dest)
    dest.mkdir(parents=True, exist_ok=True)
    record = {"git": {}, "files": {}}

    for name, (url, commit) in GIT.items():
        record["git"][name] = {"url": url, "commit": git_checkout(dest / name, url, commit)}
    for name, url in URLS.items():
        fetch(url, dest / name)
    for name, fid in GDRIVE.items():
        fetch_gdrive_zip(fid, dest / name, dest / "_downloads")

    for p in sorted(dest.rglob("*")):
        if p.is_file() and (p.suffix in {".pyth", ".csv"} or p.name.endswith(".pth.tar")) and "_downloads" not in p.parts:
            record["files"][str(p.relative_to(dest)).replace("\\", "/")] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    (dest / "ASSETS.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
