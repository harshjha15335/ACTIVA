"""Download the manifest's videos and check them. Owned by M4. Dev tooling, never used by run.sh.

Follows TEAM_README.md s2: EPIC-KITCHENS-100 via the official downloader, FPVSum
via yt-dlp using the upstream FPVSum video list (name -> YouTube ID; the "_1",
"_3" parts are separate YouTube videos). Everything lands under --out
(default data/videos, git-ignored):

    data/videos/fpvsum/<video_id>.mp4
    data/videos/EPIC-KITCHENS/P01/videos/P01_11.MP4

    python scripts/download_videos.py fpvsum            # 49 videos, small
    python scripts/download_videos.py epic              # 138 videos, ~184 GB, resumable
    python scripts/download_videos.py check             # what is there, durations vs manifest

Re-running skips videos that are already there. Every command ends with the
check, which writes data/videos/download_report.json. Then:

    python scripts/run_manifest.py --videos data/videos
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from run_manifest import index_videos  # noqa: E402
from validate_outputs import read_manifest  # noqa: E402

# Pinned upstream commits, so a change upstream can't silently change what we fetch.
FPVSUM_COMMIT = "9af31f9027ef2b3f5f34a39bf17df5b120a9cc25"
FPVSUM_LIST_URL = f"https://raw.githubusercontent.com/azuxmioy/fpvsum/{FPVSUM_COMMIT}/crawler/FPVSum_videolist.csv"
EPIC_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"
EPIC_COMMIT = "4f11fb2b579833f360c3c7bb917bf1e24a9787b5"
EPIC_GB_PER_SECOND = 184 / 47_520   # kit: ~184 GB for 13.2 h of EPIC video
# 720p is plenty for CLIP features; scoring uses seconds, not pixels.
YTDLP_FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b"


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def share(manifest, part):
    """Keep share k of n ("2/3"): every n-th video of each dataset by duration, so shares are balanced."""
    k, n = (int(x) for x in part.split("/"))
    if not 1 <= k <= n:
        raise SystemExit(f"--part must look like 1/3, got {part}")
    keep = set()
    for dataset in {ds for _, ds in manifest.values()}:
        ordered = sorted(manifest_ids(manifest, dataset), key=lambda v: manifest[v][0])
        keep.update(ordered[k - 1::n])
    return {v: row for v, row in manifest.items() if v in keep}


def manifest_ids(manifest, dataset):
    return [vid for vid, (_, ds) in manifest.items() if ds == dataset]


def probe(vid, expected, path):
    """Report entry for one expected video: ok only if it decodes and the duration matches."""
    import video_io
    entry = {"video_id": vid, "expected": expected}
    if path == "absent":
        return {**entry, "status": "missing"}
    if path is None:
        return {**entry, "status": "duplicate"}
    entry.update(path=str(path), gb=round(path.stat().st_size / 1e9, 3))
    try:
        got = video_io.inspect_video(path).duration
    except Exception as error:  # truncated or unreadable download
        return {**entry, "status": "unreadable", "error": str(error)[:200]}
    entry["duration"] = round(got, 2)
    entry["status"] = "ok" if abs(got - expected) <= config.DURATION_TOLERANCE else "duration_mismatch"
    return entry


def still_needed(manifest, out, ids):
    """IDs without a good local copy. Partial or broken files count as not downloaded."""
    have = index_videos(out) if out.exists() else {}
    return [v for v in ids if probe(v, manifest[v][0], have.get(v, "absent"))["status"] != "ok"]


# ---------------------------------------------------------------- FPVSum

def fpvsum_youtube_ids(names, cache):
    """{video_id: youtube_id} from the upstream FPVSum list (downloaded once, cached)."""
    import csv
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetching FPVSum video list ({FPVSUM_COMMIT[:7]})")
        with urllib.request.urlopen(FPVSUM_LIST_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    with open(cache, newline="", encoding="utf-8-sig") as f:
        upstream = {row["video_name"].strip(): row["youtube_id"].strip() for row in csv.DictReader(f)}
    missing = [n for n in names if n not in upstream]
    if missing:
        raise SystemExit(f"{len(missing)} manifest FPVSum IDs are not in the upstream list: {missing}")
    return {n: upstream[n] for n in names}


def download_fpvsum(manifest, out, limit=None, dry_run=False):
    names = manifest_ids(manifest, "fpvsum")[:limit]
    folder = out / "fpvsum"
    ids = fpvsum_youtube_ids(names, folder / "FPVSum_videolist.csv")
    todo = still_needed(manifest, out, names)
    log(f"FPVSum: {len(names) - len(todo)} already here, {len(todo)} to download")
    if dry_run:
        for n in todo:
            log(f"  {n} <- https://www.youtube.com/watch?v={ids[n]}")
        return
    if not shutil.which("ffmpeg"):
        log("warning: ffmpeg not on PATH; yt-dlp can't merge video+audio, falls back to single-file formats")
    failed = []
    for i, name in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {name}")
        cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress", "--retries", "10",
               "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
               "-o", str(folder / f"{name}.%(ext)s"), f"https://www.youtube.com/watch?v={ids[name]}"]
        if subprocess.run(cmd).returncode != 0:
            # Some videos list no formats to the default clients but still serve one to the
            # Android client (Scuba_Um4kqB8bQBg: 360p only, checked 2026-11-13).
            log(f"  retrying {name} with the Android client")
            retry = cmd[:-1] + ["--extractor-args", "youtube:player_client=android", cmd[-1]]
            if subprocess.run(retry).returncode != 0:
                failed.append(name)
    if failed:
        log(f"FPVSum: {len(failed)} failed (re-run to retry): {failed}")


# ---------------------------------------------------------------- EPIC

def epic_downloader(out):
    """Clone the official downloader at the pinned commit (once)."""
    repo = out / "epic-kitchens-download-scripts"
    if not (repo / "epic_downloader.py").exists():
        log("cloning the official EPIC-KITCHENS downloader")
        subprocess.run(["git", "clone", "-q", EPIC_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", EPIC_COMMIT], check=True)
    return repo


# The official downloader fetches each file over one connection. data.bris gives
# ~50-70 KB/s per connection, so a 7 GB video would take over a day. Here each
# file is split into ranges fetched by a shared pool of connections (measured:
# 16 connections ~1 MB/s total, scaling linearly), with the official URLs and MD5s.
EPIC_55_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
EPIC_100_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
CHUNK = 4 * 1024 * 1024   # ~1 min per chunk per connection at the measured speed


def epic_sources(repo, ids):
    """{video_id: (url, md5 or None)} using the official downloader's split, MD5 and errata files."""
    import csv

    def rows(name):
        with open(repo / "data" / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    known = {r["video_id"] for r in rows("epic_100_splits.csv")}
    split_55 = {r["video_id"]: r["split"] for r in rows("epic_55_splits.csv")}
    md5 = {(r["version"], r["file_remote_path"]): r["md5"] for r in rows("md5.csv")}
    errata = {r["rdsf_path"]: r["dropbox_path"] for r in rows("errata.csv")}
    unknown = [v for v in ids if v not in known]
    if unknown:
        raise SystemExit(f"not in the official downloader's list: {unknown}")
    sources = {}
    for vid in ids:
        person, number = vid.split("_")
        if len(number) == 3:   # EPIC-100 extension video (same rule as the official script)
            remote, base, version = f"{person}/videos/{vid}.MP4", EPIC_100_URL, "100"
        else:
            remote, base, version = f"videos/{split_55[vid]}/{person}/{vid}.MP4", EPIC_55_URL, "55"
        if remote in errata:
            sources[vid] = (errata[remote], md5.get(("errata", remote)))
        else:
            sources[vid] = (f"{base}/{remote}", md5.get((version, remote)))
    return sources


def epic_path(out, vid):
    return out / "EPIC-KITCHENS" / vid.split("_")[0] / "videos" / f"{vid}.MP4"


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as r:
        return int(r.headers["Content-Length"])


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RangedDownload:
    """One file: preallocated <name>.part plus <name>.part.json listing finished chunks (for resume)."""

    def __init__(self, vid, url, md5, path):
        import threading
        self.vid, self.url, self.md5, self.path = vid, url, md5, path
        self.part = path.with_suffix(".MP4.part")
        self.state = path.with_suffix(".MP4.part.json")
        self.lock = threading.Lock()
        self.size = remote_size(url)
        self.done = set()
        if self.part.exists() and self.state.exists():
            saved = json.loads(self.state.read_text())
            if (saved.get("url"), saved.get("size"), saved.get("chunk")) == (url, self.size, CHUNK):
                self.done = set(saved["done"])
        if not self.done:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.part, "wb") as f:
                f.truncate(self.size)
        self.chunks = [i for i in range((self.size + CHUNK - 1) // CHUNK) if i not in self.done]
        self.left = len(self.chunks)

    def fetch(self, index, stop):
        start = index * CHUNK
        end = min(start + CHUNK, self.size) - 1
        for attempt in range(10):
            if stop.is_set():
                return 0
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as r:
                    if r.status != 206:
                        raise OSError(f"server ignored the range request (HTTP {r.status})")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read {len(data)} of {end - start + 1}")
                with open(self.part, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                with self.lock:
                    self.done.add(index)
                    self.left -= 1
                    self.state.write_text(json.dumps({"url": self.url, "size": self.size, "chunk": CHUNK,
                                                      "done": sorted(self.done)}))
                    finished = self.left == 0
                if finished:
                    self.finish()
                return len(data)
            except Exception as error:
                if attempt == 9:
                    log(f"  {self.vid} chunk {index} failed: {error}")
                    return 0
                time.sleep(min(60, 2 ** attempt))
        return 0

    def finish(self):
        if self.md5 and file_md5(self.part) != self.md5:
            log(f"  {self.vid}: MD5 mismatch, deleting it (re-run to fetch again)")
            self.part.unlink()
            self.state.unlink()
            return
        self.part.replace(self.path)
        self.state.unlink()
        log(f"  done {self.vid} ({self.size / 1e9:.2f} GB, MD5 {'ok' if self.md5 else 'not listed'})")


def download_epic(manifest, out, connections=32, limit=None, dry_run=False):
    import concurrent.futures
    import threading
    # Shortest videos first, so the batch run can start on early ones.
    all_ids = manifest_ids(manifest, "epic_kitchens_100")
    ids = sorted(all_ids[:limit], key=lambda v: manifest[v][0])
    todo = still_needed(manifest, out, ids)
    log(f"EPIC: {len(ids) - len(todo)} already here, {len(todo)} to download")
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    sources = epic_sources(epic_downloader(out), todo)
    free_gb = shutil.disk_usage(out).free / 1e9
    need_gb = EPIC_GB_PER_SECOND * sum(manifest[v][0] for v in todo)
    log(f"disk: {free_gb:.0f} GB free, roughly {need_gb:.0f} GB needed")
    if dry_run:
        for v in todo:
            log(f"  {v} <- {sources[v][0]}")
        return
    if free_gb < need_gb * 1.05:
        raise SystemExit("not enough free disk space")

    for v in todo:   # a finished file that failed the check is replaced, never trusted
        if epic_path(out, v).exists():
            epic_path(out, v).unlink()
    downloads = [RangedDownload(v, *sources[v], epic_path(out, v)) for v in todo]
    total = sum(d.size for d in downloads)
    left = sum(d.left for d in downloads) * CHUNK
    log(f"{len(downloads)} files, {total / 1e9:.1f} GB ({max(total - left, 0) / 1e9:.1f} GB already fetched), "
        f"{connections} connections")
    stop = threading.Event()
    got = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=connections)
    futures = [pool.submit(d.fetch, i, stop) for d in downloads for i in d.chunks]
    started = last = time.time()
    try:
        for future in concurrent.futures.as_completed(futures):
            got += future.result() or 0
            if time.time() - last >= 60:
                last = time.time()
                rate = got / (last - started)
                left = sum(d.left for d in downloads) * CHUNK
                finished = sum(1 for d in downloads if d.left == 0)
                log(f"  {got / 1e9:.2f} GB this run, {rate / 1e6:.2f} MB/s, {finished}/{len(downloads)} files, "
                    f"ETA {left / max(rate, 1) / 3600:.1f} h")
    except KeyboardInterrupt:
        log("stopping; progress is saved, re-run the same command to resume")
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown()


# ---------------------------------------------------------------- check

def check(manifest, out, datasets):
    """Probe every expected video: present, decodable, duration within tolerance."""
    have = index_videos(out) if out.exists() else {}
    entries = [{"dataset": ds, **probe(vid, expected, have.get(vid, "absent"))}
               for vid, (expected, ds) in manifest.items() if ds in datasets]
    summary = {}
    for e in entries:
        summary[e["status"]] = summary.get(e["status"], 0) + 1
    report = {"summary": summary, "total_gb": round(sum(e.get("gb", 0) for e in entries), 1), "entries": entries}
    out.mkdir(parents=True, exist_ok=True)
    (out / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{summary}  ({report['total_gb']} GB)  -> {out / 'download_report.json'}")
    for e in entries:
        if e["status"] != "ok":
            print(f"  {e['status']:<18} {e['video_id']}" + (f"  got {e['duration']} vs {e['expected']}"
                                                           if "duration" in e else ""))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["fpvsum", "epic", "check"])
    parser.add_argument("--out", default=str(ROOT / "data" / "videos"))
    parser.add_argument("--manifest", default=str(config.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--connections", type=int, default=32,
                        help="parallel EPIC connections (default 32; ~60 KB/s each, so 32 fills a ~2 MB/s line)")
    parser.add_argument("--limit", type=int, help="only the first N manifest videos of that dataset")
    parser.add_argument("--part", help="only share k of n, e.g. 1/3, to split the download across people")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = parser.parse_args(argv)

    manifest = read_manifest(args.manifest)
    if args.part:
        manifest = share(manifest, args.part)
    out = Path(args.out)
    if args.what == "fpvsum":
        download_fpvsum(manifest, out, args.limit, args.dry_run)
    elif args.what == "epic":
        download_epic(manifest, out, max(1, args.connections), args.limit, args.dry_run)
    if args.dry_run:
        return 0
    datasets = {"fpvsum": ["fpvsum"], "epic": ["epic_kitchens_100"]}.get(args.what, ["fpvsum", "epic_kitchens_100"])
    if args.limit:
        keep = set(manifest_ids(manifest, datasets[0])[:args.limit])
        manifest = {k: v for k, v in manifest.items() if k in keep}
    report = check(manifest, out, datasets)
    return 0 if set(report["summary"]) <= {"ok"} else 1


if __name__ == "__main__":
    sys.exit(main())
