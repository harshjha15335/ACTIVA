"""ACTIVA local spec validator. Owned by M4.

NOT the official validate.py: the student kit's README lists one, but the ZIP we
received does not contain it. This checks our reading of TEAM_README.md s3/s4
(see code/schema.py) plus the submission-level rules:

- every file is <video_id>.json and its video_id matches the file name
- with a manifest: duration_sec within +-0.5 s, every manifest video present,
  no files for videos outside the manifest
- team_id identical across all files

    python scripts/validate_outputs.py --file outputs/P01_11.json -v
    python scripts/validate_outputs.py --dir outputs/ [--manifest kit/manifest.csv]

Exit code 0 if every file is valid (warnings allowed), 1 otherwise.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import config  # noqa: E402
import schema  # noqa: E402
import vocab as vocab_module  # noqa: E402


def read_manifest(path):
    """{video_id: (duration_sec, dataset)} from the kit's manifest.csv."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"].strip()
            if vid in rows:
                raise ValueError(f"{path}: duplicate video_id {vid}")
            rows[vid] = (float(row["duration_sec"]), row["dataset"].strip())
    return rows


def check_file(path, vocab, manifest=None):
    """Return (data or None, errors, warnings) for one output file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, [f"cannot read JSON: {error}"], []
    errors, warnings = [], []
    expected = None
    if isinstance(data, dict):
        if data.get("video_id") != path.stem:
            errors.append(f"video_id {data.get('video_id')!r} does not match file name {path.stem!r}")
        if manifest is not None:
            if path.stem in manifest:
                expected = manifest[path.stem][0]
            else:
                errors.append(f"{path.stem} is not in the manifest")
    try:
        warnings = schema.validate_output(data, expected, vocab)
    except schema.SchemaError as error:
        errors.extend(str(error).split("; "))
    return data, errors, warnings


def validate_dir(directory, vocab, manifest=None):
    """Return {name: (errors, warnings)} plus a "<submission>" entry for cross-file problems."""
    results, team_ids = {}, {}
    files = sorted(Path(directory).glob("*.json"))
    for path in files:
        data, errors, warnings = check_file(path, vocab, manifest)
        results[path.name] = (errors, warnings)
        if isinstance(data, dict) and isinstance(data.get("team_id"), str):
            team_ids.setdefault(data["team_id"], []).append(path.name)
    problems = []
    if len(team_ids) > 1:
        problems.append(f"team_id differs across files: { {k: len(v) for k, v in team_ids.items()} }")
    if manifest is not None:
        missing = sorted(set(manifest) - {p.stem for p in files})
        if missing:
            problems.append(f"{len(missing)} manifest videos have no output: {missing[:10]}"
                            + (" ..." if len(missing) > 10 else ""))
    if not files:
        problems.append(f"no .json files in {directory}")
    results["<submission>"] = (problems, [])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACTIVA local spec validator (not the official validate.py)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file")
    target.add_argument("--dir")
    parser.add_argument("--manifest", help="kit manifest.csv (default: kit/manifest.csv if present)")
    parser.add_argument("--vocab", help="supported_classes.json (default: config.vocab_path())")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print warnings")
    args = parser.parse_args(argv)

    vocab = vocab_module.load(args.vocab)
    manifest_path = args.manifest or (config.DEFAULT_MANIFEST_PATH if config.DEFAULT_MANIFEST_PATH.exists() else None)
    manifest = read_manifest(manifest_path) if manifest_path else None
    print(f"ACTIVA local spec validator (official validate.py not available); "
          f"manifest: {manifest_path or 'none'}")

    if args.file:
        _, errors, warnings = check_file(args.file, vocab, manifest)
        results = {Path(args.file).name: (errors, warnings)}
    else:
        results = validate_dir(args.dir, vocab, manifest)

    failed = 0
    for name, (errors, warnings) in results.items():
        if errors:
            failed += name != "<submission>"
            print(f"FAIL {name}")
            for e in errors:
                print(f"     - {e}")
        elif name != "<submission>":
            print(f"OK   {name}" + (f" ({len(warnings)} warnings)" if warnings else ""))
        if args.verbose:
            for w in warnings:
                print(f"     ~ {w}")
    files = len([n for n in results if n != "<submission>"])
    cross = results.get("<submission>", ([], []))[0]
    print(f"{files - failed}/{files} files valid" + (f"; {len(cross)} submission-level problems" if cross else ""))
    return 1 if failed or cross else 0


if __name__ == "__main__":
    sys.exit(main())
