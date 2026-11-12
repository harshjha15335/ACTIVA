"""Run the pipeline on every manifest video. Owned by M4.

    python scripts/run_manifest.py --videos /data/videos [--out outputs] [--manifest kit/manifest.csv]
                                   [--dataset epic_kitchens_100|fpvsum] [--limit N] [--force]

- Finds each manifest video by file name: <video_id>.<ext> anywhere under --videos
  (ext: mp4, MP4, mkv, webm, mov). The file name must equal the manifest video_id.
- Resumable: an existing output that already passes the local spec validator is
  skipped unless --force.
- Calls the same code/main.py run() as ./run.sh, in one process so models load once.
- Writes a status report after every video (default tmp/run_report.json, not in
  outputs/, which must hold only the 187 submission files).

Exit code 0 if every selected video has a valid output, 1 otherwise.
Set ACTIVA_STRICT=1 for a release run where any stage fallback counts as a failure.
"""
import argparse
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import main as pipeline  # noqa: E402
import schema  # noqa: E402
import video_io  # noqa: E402
import vocab as vocab_module  # noqa: E402
from validate_outputs import check_file, read_manifest  # noqa: E402

log = logging.getLogger("activa.batch")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}


def index_videos(directory):
    """{stem: path} for video files under directory; stems seen twice map to None."""
    found = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            found[path.stem] = None if path.stem in found else path
    return found


def run_one(video_id, path, expected_duration, out_path, vocab, team):
    """Process one video; return a report entry."""
    entry = {"video_id": video_id, "video": str(path), "status": None}
    started = time.perf_counter()
    try:
        result = pipeline.run(path)
        out = result.output
        entry["fallbacks"] = result.fallbacks
        entry["timings"] = {k: round(v, 2) for k, v in result.timings.items()}
        entry["warnings"] = schema.validate_output(out, expected_duration, vocab, team_id=team)
        pipeline.write_json_atomic(out, out_path)
        entry["status"] = "ok" if not result.fallbacks else "ok_with_fallbacks"
    except video_io.VideoError as error:
        entry.update(status="invalid_video", error=str(error))
    except pipeline.StageError as error:
        entry.update(status="strict_failure", error=str(error))
    except schema.SchemaError as error:
        entry.update(status="invalid_output", error=str(error))
    except Exception as error:  # keep going: one bad video must not stop the batch
        log.exception("internal error on %s", video_id)
        entry.update(status="internal_error", error=f"{type(error).__name__}: {error}")
    entry["seconds"] = round(time.perf_counter() - started, 2)
    return entry


def run_manifest(videos, out_dir, manifest_path, report_path, dataset=None, limit=None, force=False):
    manifest = read_manifest(manifest_path)
    ids = [vid for vid, (_, ds) in manifest.items() if dataset in (None, ds)]
    if limit is not None:
        ids = ids[:limit]
    vocab = vocab_module.load()
    team = config.team_id()
    index = index_videos(videos)
    out_dir, report_path = Path(out_dir), Path(report_path)
    report = {"manifest": str(manifest_path), "videos_dir": str(videos), "out_dir": str(out_dir),
              "team_id": team, "strict": config.strict(), "entries": []}
    started = time.perf_counter()

    for n, vid in enumerate(ids, 1):
        duration, ds = manifest[vid]
        out_path = out_dir / f"{vid}.json"
        if out_path.exists() and not force:
            _, errors, _ = check_file(out_path, vocab, manifest)
            if not errors:
                entry = {"video_id": vid, "status": "skipped_valid"}
                report["entries"].append(entry | {"dataset": ds})
                log.info("[%d/%d] %s: already valid, skipped", n, len(ids), vid)
                continue
        path = index.get(vid)
        if vid in index and path is None:
            entry = {"video_id": vid, "status": "ambiguous_video", "error": "several files with this name"}
        elif path is None:
            entry = {"video_id": vid, "status": "missing_video"}
        else:
            entry = run_one(vid, path, duration, out_path, vocab, team)
        entry["dataset"] = ds
        report["entries"].append(entry)
        log.info("[%d/%d] %s: %s (%.1fs)%s", n, len(ids), vid, entry["status"], entry.get("seconds", 0.0),
                 f" - {entry['error']}" if "error" in entry else "")
        _write_report(report, report_path, started)

    _write_report(report, report_path, started)
    return report


def _write_report(report, path, started):
    counts = {}
    for e in report["entries"]:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    report["summary"] = counts
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    pipeline.write_json_atomic(report, path)


GOOD = {"ok", "ok_with_fallbacks", "skipped_valid"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", required=True, help="folder containing the manifest videos (searched recursively)")
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--manifest", default=str(config.DEFAULT_MANIFEST_PATH))
    parser.add_argument("--report", default=str(ROOT / "tmp" / "run_report.json"))
    parser.add_argument("--dataset", choices=["epic_kitchens_100", "fpvsum"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="re-run videos that already have a valid output")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=config.LOG_LEVEL, format="[activa] %(levelname)s %(message)s")
    try:
        report = run_manifest(args.videos, args.out, args.manifest, args.report,
                              args.dataset, args.limit, args.force)
    except (vocab_module.VocabError, config.ConfigError, FileNotFoundError) as error:
        log.error("configuration error: %s", error)
        return pipeline.EXIT_CONFIG
    log.info("summary: %s; report: %s", report["summary"], args.report)
    return 0 if all(e["status"] in GOOD for e in report["entries"]) else 1


if __name__ == "__main__":
    sys.exit(main())
