# M2 leakage audit — unresolved

No videos are approved for training or tuning. No datasets, evaluation manifest,
student kit or official vocabulary have been supplied. The perception guide is
an engineering reference; its organizer and dataset claims remain unverified.

## Required evidence

Obtain the official evaluation inventory and the public FPVSum inventory. Inspect
the authoritative annotation README before parsing human scores. Match IDs,
filenames and hashes; review decoded-content fingerprints, duration and frames
to investigate renamed or re-encoded clips. Missing exact matches do not establish
safety. Classify every clip as evaluation-overlap, reviewed-safe or unresolved,
with evidence. A teammate must review the safe list before training/tuning.
Split by video, freeze folds and exclude both overlap and unresolved clips.

`scripts/audit_ids.py` accepts our explicitly defined preliminary inventory CSV
format: `video_id,sha256`. It flags exact ID/hash overlap and leaves every other
clip unresolved. It never emits a safe training list. This is not a parser for an
assumed Samsung manifest, nor a completed content-overlap audit.

## Independent implementation status

The initial stubs were replaced after the user authorized building the parts that
can be developed independently. Guide A5 contracts and a decoder were bootstrapped
locally to unblock M2; M4 integration review is still needed. Offline CLIP
extraction, configurable category prediction, four importance baselines, cache
export and an untrained TCN architecture are implemented. The motion-only CLI
runs without weights. See README.md and docs/results.md for usage and evidence.

No supervised training, dataset tuning or benchmark evaluation has occurred.
There are no claimed F1/accuracy results, approved safe videos or final model
selection. No message-board posts, remote pushes, merges or submissions occurred.
