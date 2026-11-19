"""One-time setup: install every model file run.sh needs, and verify each SHA-256. Owned by M4.

    python scripts/setup_weights.py            # join the pieces in weights/parts/ (no network)
    python scripts/setup_weights.py --offline  # the same, but never fall back to a download
    python scripts/setup_weights.py --check    # only verify what is already in weights/
    python scripts/setup_weights.py --upstream # from the original sources instead

The weights are committed to this repo, as the organisers advised (#46). GitHub refuses
files over 100 MiB, so each file is stored as numbered pieces, weights/parts/<asset>.000,
.001, ... (95 MiB each), and joined here; the joined file must match its SHA-256. If the
pieces are missing or don't match, the file is downloaded from the repo's `weights-v1`
release instead (same bytes), unless --offline is given. Files already present with the
right hash are kept, and an interrupted download resumes. ./run.sh itself never downloads
anything. The files are the upstream ones, unchanged (licences: README); `--upstream`
rebuilds the same layout from the original authors with scripts/setup_m1_assets.py and
M2's scripts/prepare_weights.py. Maintainers: `--make-parts` re-cuts weights/parts/ from
the installed, verified files.

Layout written (what code/config.py and code/action_config.py expect):
  weights/ViT-B-32-laion400m_e32.safetensors (+ .manifest.json)   CLIP (M2)
  weights/m1/SlowFast.pyth, epic_slowfast_{verb,noun}_reproduce/epoch_020.pth.tar,
  weights/m1/epic-kitchens-slowfast/, actionformer_release/, EPIC_100_{verb,noun}_classes.csv  (M1)
"""
import argparse
import glob
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/harshjha15335/ACTIVA/releases/download/weights-v1"
PARTS_DIR = "weights/parts"     # committed pieces, relative to ROOT
PART_BYTES = 95 * 2**20         # under GitHub's 100 MiB per-file limit

# asset name -> (destination under the repo, bytes, sha256)
ASSETS = {
    "ViT-B-32-laion400m_e32.safetensors": (
        "weights/ViT-B-32-laion400m_e32.safetensors", 605143452,
        "c2ea63325cebca85a9eb509481f37ea991cb20199cab6642e68dbc8f973382f5"),
    "ViT-B-32-laion400m_e32.manifest.json": (
        "weights/ViT-B-32-laion400m_e32.manifest.json", None,
        "998828ecfa4026983ac7e6ffa96092edd6241233ce107f6086a80d45ff4efe79"),
    "SlowFast.pyth": (
        "weights/m1/SlowFast.pyth", 276771081,
        "0fc9e83d544544617471835bd345aa71a6b75d979124a492b793b786f0e90cf7"),
    "epic_slowfast_verb_reproduce.epoch_020.pth.tar": (
        "weights/m1/epic_slowfast_verb_reproduce/epoch_020.pth.tar", 476529049,
        "88486434a71e1b17ac38dd4a93d9612612026be62d7a93a032a0953a30ac687b"),
    "epic_slowfast_noun_reproduce.epoch_020.pth.tar": (
        "weights/m1/epic_slowfast_noun_reproduce/epoch_020.pth.tar", 481521049,
        "8d313a2dd69562bfb063cb1e3372ad056ee646283d1e32af95ff49318bf6bbf9"),
}
# Model code + class lists, unpacked into weights/m1/ (checked by its zip hash and a marker file).
CODE_ZIP = ("m1_code.zip", "weights/m1", "d60aba99f3531715599b75e74688483f21c6f5eee8e620994a1c05fc5a7f7000")
CODE_MARKERS = ["epic-kitchens-slowfast/configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml",
                "actionformer_release/configs/epic_slowfast_verb.yaml",
                "actionformer_release/configs/epic_slowfast_noun.yaml",
                "EPIC_100_verb_classes.csv", "EPIC_100_noun_classes.csv"]


def log(message):
    print(message, file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size=None):
    """Download url to dest via dest.part, resuming a partial file when the server allows."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "activa-setup"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:
            have = 0   # no resume support: start over
        total = (int(response.headers.get("Content-Length", 0)) + have) or size
        with open(part, "ab" if have else "wb") as out:
            done, shown = have, -1
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total and (pct := done * 100 // total) // 10 != shown:
                    shown = pct // 10
                    log(f"  {dest.name}: {pct}% of {total / 1e6:.0f} MB")
    part.replace(dest)


def ok(path, want):
    return path.is_file() and sha256(path) == want


def parts_of(asset):
    """The committed pieces of asset, in order: weights/parts/<asset>.000, .001, ..."""
    return sorted((ROOT / PARTS_DIR).glob(glob.escape(asset) + ".[0-9][0-9][0-9]"))


def join_parts(asset, dest, want):
    """Rebuild dest from its committed pieces. True if the result matches want."""
    parts = parts_of(asset)
    if not parts:
        return False
    if [p.name for p in parts] != [f"{asset}.{i:03d}" for i in range(len(parts))]:
        log(f"  {asset}: pieces in {PARTS_DIR}/ are not numbered .000 to .{len(parts) - 1:03d}; not used")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    joining = dest.with_name(dest.name + ".joining")
    with open(joining, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
    if sha256(joining) != want:
        joining.unlink()
        log(f"  {asset}: the {len(parts)} pieces in {PARTS_DIR}/ don't match the SHA-256; not used")
        return False
    joining.replace(dest)
    log(f"join    {dest.relative_to(ROOT)} from {len(parts)} pieces in {PARTS_DIR}/")
    return True


def make_parts(asset, source, want):
    """Cut a verified file into PART_BYTES pieces in weights/parts/ (maintainers only)."""
    if not ok(source, want):
        sys.exit(f"{source.relative_to(ROOT)} is missing or has the wrong hash; install it first")
    out_dir = ROOT / PARTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_of(asset):
        old.unlink()
    n = 0
    with open(source, "rb") as f:
        while block := f.read(PART_BYTES):
            (out_dir / f"{asset}.{n:03d}").write_bytes(block)
            n += 1
    log(f"cut     {asset} into {n} pieces")


def fetch(asset, dest, size, want, base, offline=False):
    if ok(dest, want):
        log(f"ok      {dest.relative_to(ROOT)}")
        return
    if dest.exists():
        log(f"replace {dest.relative_to(ROOT)} (wrong hash)")
        dest.unlink()
    if join_parts(asset, dest, want):
        return
    if offline:
        sys.exit(f"{asset}: no usable pieces in {PARTS_DIR}/ and --offline was given")
    log(f"get     {asset} (from the release)")
    for attempt in range(1, 6):
        try:
            download(f"{base}/{asset}", dest, size)
            break
        except OSError as error:
            log(f"  attempt {attempt} failed: {error}")
            if attempt == 5:
                raise
    got = sha256(dest)
    if got != want:
        dest.unlink()
        sys.exit(f"SHA-256 mismatch for {asset}: got {got}, expected {want}. Deleted; re-run to retry.")


def code_archive():
    name, dest_dir, _ = CODE_ZIP
    return ROOT / dest_dir / "_downloads" / name


def install_code(base, offline=False):
    name, dest_dir, want = CODE_ZIP
    dest_dir = ROOT / dest_dir
    if all((dest_dir / marker).is_file() for marker in CODE_MARKERS):
        log(f"ok      {dest_dir.relative_to(ROOT)}/ model code")
        return
    archive = code_archive()
    fetch(name, archive, None, want, base, offline)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():   # refuse paths that escape weights/m1
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                sys.exit(f"unsafe path in {name}: {member}")
        # The third-party source is committed (D-d); never overwrite it, or a fresh clone
        # shows every vendored file as modified (the zip has CRLF endings). Only the
        # missing members (the EPIC class CSVs) are written.
        added = 0
        for member in z.infolist():
            if member.is_dir() or (dest_dir / member.filename).exists():
                continue
            z.extract(member, dest_dir)
            added += 1
    log(f"unpack  {name} -> {dest_dir.relative_to(ROOT)}/ ({added} new files)")


def check():
    """Return the list of problems with the installed files (empty means ready)."""
    bad = [f"{dest}: {'wrong hash' if (ROOT / dest).exists() else 'missing'}"
           for dest, _, want in ASSETS.values() if not ok(ROOT / dest, want)]
    bad += [f"weights/m1/{m}: missing" for m in CODE_MARKERS if not (ROOT / "weights" / "m1" / m).is_file()]
    return bad


def upstream():
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "setup_m1_assets.py")], check=True)
    subprocess.run([py, str(ROOT / "scripts" / "prepare_weights.py"), "--preset", "laion400m_e32",
                    "--output", str(ROOT / ASSETS["ViT-B-32-laion400m_e32.safetensors"][0])], check=True)
    # The converted CLIP file's bytes differ by machine, so only the M1 files are hash-checked here.
    return [b for b in check() if not b.startswith("weights/ViT-B-32")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="only verify the installed files")
    mode.add_argument("--upstream", action="store_true", help="download from the original sources")
    mode.add_argument("--offline", action="store_true",
                      help=f"only join the committed pieces in {PARTS_DIR}/; never download")
    mode.add_argument("--make-parts", action="store_true",
                      help=f"maintainers: re-cut {PARTS_DIR}/ from the installed, verified files")
    ap.add_argument("--base-url", default=os.environ.get("ACTIVA_WEIGHTS_URL", RELEASE),
                    help="where the release assets live (default: this repo's weights-v1 release)")
    args = ap.parse_args(argv)
    if args.upstream:
        bad = upstream()
    elif args.check:
        bad = check()
    elif args.make_parts:
        for asset, (dest, _, want) in ASSETS.items():
            make_parts(asset, ROOT / dest, want)
        make_parts(CODE_ZIP[0], code_archive(), CODE_ZIP[2])
        return 0
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(ROOT).free
        if free < 2.5e9:
            log(f"warning: {free / 1e9:.1f} GB free; about 1.9 GB is needed")
        for asset, (dest, size, want) in ASSETS.items():
            fetch(asset, ROOT / dest, size, want, args.base_url, args.offline)
        install_code(args.base_url, args.offline)
        bad = check()
    if bad:
        log("NOT READY:\n  " + "\n  ".join(bad))
        return 1
    log("all model files present and verified; ./run.sh can run offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
