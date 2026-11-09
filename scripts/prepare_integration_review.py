#!/usr/bin/env python3
"""Prepare a review-only M2 patch and optional team rehearsal from immutable refs.

No branch switch, history merge, remote write or canonical checkout modification.
The output directory must be new. Actual owner files stay byte-for-byte unchanged.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def materialize(ref, destination):
    commit = git('rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip()
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(git('archive', commit)), mode='r:') as archive:
        archive.extractall(destination, filter='data')
    return commit


def copy_owned(source, target):
    copied = []
    for directory, patterns in {
        'code': ['clip_features.py', 'importance.py', 'category.py', 'category_vocab.py',
                 'clip_backend.py', 'tcn.py', 'perception_*.py', 'fpvsum_*.py', 'leakage_*.py'],
        'scripts': ['*.py'], 'tests': ['test_*.py'],
        'weights': ['README_weights.md'],
        'docs': ['M2_*.md', 'results.md', 'leakage.md', 'evaluation_formats.md', 'evidence/*.json'],
    }.items():
        for pattern in patterns:
            for path in (source / directory).glob(pattern):
                relative = path.relative_to(source)
                out = target / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out)
                copied.append(str(relative))
    # Preserve canonical README/decisions and add M2's material separately.
    shutil.copyfile(source / 'README.md', target / 'docs' / 'M2_README.md')
    with (target / 'docs' / 'decisions.md').open('a') as stream:
        stream.write('\n\n## M2 decisions proposed for review\n\n')
        stream.write((source / 'docs' / 'decisions.md').read_text())
    for name in ['requirements.txt', 'requirements-dev.txt']:
        original = (target / name).read_text()
        seen = {line.strip() for line in original.splitlines()}
        additions = [line for line in (source / name).read_text().splitlines()
                     if line.strip() and not line.startswith('#') and line.strip() not in seen]
        (target / name).write_text(original.rstrip() + '\n' + '\n'.join(additions) + '\n')
    shutil.copyfile(source / 'requirements-m2.txt', target / 'requirements-m2.txt')
    shutil.copyfile(source / 'perceive.sh', target / 'perceive.sh')
    (target / 'perceive.sh').chmod(0o755)
    return sorted(set(copied))


def overlay_ref(ref, destination, paths):
    commit = git('rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip()
    files = git('ls-tree', '-r', '--name-only', commit).decode().splitlines()
    copied = []
    for name in files:
        if any(name == path or name.startswith(path.rstrip('/') + '/') for path in paths):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git('show', f'{commit}:{name}'))
            copied.append(name)
    return {'commit': commit, 'paths': copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/main')
    parser.add_argument('--output', required=True)
    parser.add_argument('--m1-ref')
    parser.add_argument('--m3-ref')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('Output must be a new directory; existing work is never overwritten.')
    output.mkdir(parents=True)
    canonical, review = output / 'canonical', output / 'review'
    commit = materialize(args.base, canonical)
    shutil.copytree(canonical, review)
    owned = copy_owned(ROOT, review)
    protected = ['code/contracts.py', 'code/config.py', 'code/vocab.py', 'code/video_io.py',
                 'code/main.py', 'code/schema.py', 'run.sh', 'AGENTS.md']
    if any((canonical / name).read_bytes() != (review / name).read_bytes() for name in protected):
        raise RuntimeError('Protected owner files unexpectedly changed.')
    diff = subprocess.run(['git', 'diff', '--no-index', '--no-renames', '--src-prefix=a/',
                           '--dst-prefix=b/', 'canonical', 'review'], cwd=output,
                          capture_output=True, text=True)
    if diff.returncode not in (0, 1):
        raise RuntimeError(diff.stderr)
    lines = []
    for line in diff.stdout.splitlines(keepends=True):
        if line.startswith(('diff --git ', '--- ', '+++ ')):
            line = line.replace('a/canonical/', 'a/').replace('b/review/', 'b/')
            # Added/deleted files can use either directory on both diff headers.
            line = line.replace('a/review/', 'a/').replace('b/canonical/', 'b/')
        lines.append(line)
    patch = output / 'm2-to-main.patch'
    patch.write_text(''.join(lines))
    check = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=canonical,
                           capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError('Review patch failed clean-base apply check: ' + check.stderr)
    record = {'format': 'activa-m2-integration-review-v1', 'canonical_base': commit,
              'M2_source_commit': git('rev-parse', 'HEAD').decode().strip(),
              'M2_worktree_used': True, 'copied_M2_paths': owned,
              'protected_files_unchanged': protected, 'patch_apply_check': 'PASS',
              'official_vocabulary_verified': False, 'no_history_merge_performed': True}
    if args.m1_ref or args.m3_ref:
        combined = output / 'combined'
        shutil.copytree(review, combined)
        if args.m1_ref:
            record['M1'] = overlay_ref(args.m1_ref, combined,
                ['code/actions.py', 'code/action_config.py', 'tests/test_actions.py', 'tests/conftest.py'])
        if args.m3_ref:
            record['M3'] = overlay_ref(args.m3_ref, combined,
                ['code/optimizer.py', 'eval', 'tests/test_budget.py', 'tests/test_intervals.py',
                 'tests/test_metrics.py', 'requirements-m3.txt'])
    (output / 'review-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
