# FPVSum leakage status: BLOCKED BY MANIFEST

**No video IDs are approved for training or tuning.** No supervised training,
protected evaluation tuning, held-out quality evaluation or model-selection run
was performed during this remediation.

## Dataset version and verified source

Upstream: github.com/azuxmioy/fpvsum, commit
9af31f9027ef2b3f5f34a39bf17df5b120a9cc25. Audit/remediation inspected the actual CSV
and MAT files:98 videos,14 categories,56 labeled,42 unlabeled;56 processed-score records
and978 raw annotator records. README's approximate35 percent statement does not
replace these file counts. Public availability does not imply evaluation safety.

Official Samsung manifest availability: **missing**. Official vocabulary/evaluator
also unavailable. Review status: **not reviewed/cleared**.

## Explicit unresolved discrepancies

- Processed/raw Name `Jumping_mU-p2wuk90` vs CSV `Jumping__mU-p2wuk90`.
  A spelling similarity is not verified source-video identity. No alias applied.
- Raw category `BikePolo` vs CSV `Bikepolo` for four known labeled IDs.
  No automatic case conversion is applied; an evidence-backed category alias can
  be supplied explicitly to the loader after human review.
- Some raw Label arrays are fractional finite[0,1] (160 of 978 records), not binary.
  The loader preserves them as float64. Processed Score is not generally reproduced
  by naive raw averaging. That aggregation/alignment method remains unverified.
- Annotation files do not supply the original decoded PTS grid. Resampling requires
  an explicit score-aligned source timestamp vector, verified against the reviewed
  original video. Never infer correctness from a redownload's nominalFPS alone.
- Upstream repository has no explicit dataset LICENSE file; reuse obligations need
  checking. No dataset videos/annotations are committed here.

## Matching and split methodology

1. Collect official manifest and authorized public-video identity evidence.
2. Match exact IDs/raw hashes/source identifiers. Generate first-seconds decoded
   256-bit difference-hash fingerprints plus duration/aspect metadata for human
   re-encoding/rename investigation. Similarity is only a candidate; flat footage
   can collide, and shifted/trimmed copies can be missed.
3. Record each as reviewed-safe,evaluation-overlap or unresolved with evidence,
   exact file hash and human reviewer. Bind the review to the actual manifest hash.
4. Exclude overlap and unresolved videos. Keep original/source-content groups and
   identical hashes out of different folds; split by video/source, not frame.
5. Have another teammate review before any training,tuning or baseline selection.

Safe training IDs: **none approved**.
Excluded evaluation IDs: **unknown until official manifest arrives**.
Unresolved: **all public FPVSum videos for training purposes**.

## Implemented tools

- fpvsum_data.py: strict processed/raw loaders, reviewed aliases, explicit duration-
  weighted score-bin alignment and manifest-bound review/split/hash guards.
- fingerprint_videos.py: candidate fingerprints only.
- audit_ids.py: preliminary own-format video_id,sha256 CSV; optional fingerprint
  inventories add unresolved content candidates. It never grants training clearance.
- evaluate_perception.py: fails before model use if reviewed decisions/official
  manifest binding or video hashes are missing. Its metrics are development proxies.

See evaluation_formats.md. These are ACTIVA adapter schemas, not invented Samsung
manifest formats. M4 must supply the real manifest adapter/interpretation.
Production perception never loads these annotations,reviews or development caches.

## Current integration check — 9 November 2026

Fresh source inspection reconfirmed 98 videos / 56 labeled / 42 unlabeled and the same
strict parser ID/category blockers. Canonical main now exists, but its kit folder
contains instructions only; its vocabulary fixture is explicitly not official.
No official manifest or newly approved safe list was found. Status remains
BLOCKED BY MANIFEST and no training or held-out model-quality run occurred.
