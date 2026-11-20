# M2 development data formats and safety gates

These are explicit local adapter formats, **not official Samsung file formats**.
The canonical shared dataclasses are unchanged. Production inference does not read
annotations,training lists or caches. No training script is enabled or run.

## Upstream FPVSum

Inventory CSV columns: video_name,youtube_id,category,split(labeled/unlabeled).
MAT gt_data records: Name and normalized Score; the additional upstream fields
are preserved in the original MAT source but not treated as judge ground truth.
MAT raw_data: Name,Category,Annotator_id,Label. Preserve fractional Label vectors;
no undocumented aggregation is performed. Unknown IDs,categories,nonfinite/range
errors,missing required processed records and duplicate annotators fail clearly.

Optional reviewed alias JSON:

```json
{"format":"activa-reviewed-id-aliases-v1","reviewed_by":"ACTUAL_REVIEWER","evidence":"ACTUAL_CONTENT_EVIDENCE","aliases":{},"category_aliases":{}}
```

Do not populate candidate underscore/case mappings without verification. The actual
upstream identity discrepancy remains unresolved. RawAnnotation retains original
category text even when an explicitly reviewed alias validates its membership.

## Annotation resampling

Supply NPZ `times` with one original decoded PTS per processed Score frame, strict
increasing seconds and known reviewed source duration. Scores are treated as
piecewise constant on[source_pts[i],source_pts[i+1]); final frame ends at duration.
Target bins are[clip_time[i],clip_time[i+1]); final bin ends at duration. Target
values are duration-weighted means, bounded[0,1]. Length/coverage mismatch fails;
no FPS guess is made. This is our documented development convention; official
label alignment still needs M3/M4 approval.

## Human review decision

```json
{"format":"activa-leakage-review-v1","state":"pending","reviewed_by":"","reviewed_at":"","official_manifest_sha256":"","videos":[{"video_id":"EXPLICIT_ID","status":"unresolved"}]}
```

This pending template is intentionally rejected by evaluation. A real review must
be state reviewed, name reviewer/time, match the SHA256 of the actual official
manifest file and contain unique decisions. Reviewed-safe records require sha256
and evidence; optional source_group links known re-encodings. Human attestation
is not automatically inferred or independently authenticated by this file format.
No such reviewed decision exists in this repository.

assert_safe_split rejects unreviewed/unlabeled/overlap IDs and cross-fold video,
source ID,reviewed source_group or exact-content duplicates. assert_video_hash
binds the evaluated file to the reviewed decision. The code cannot magically
establish that an arbitrary supplied file is the official organizer manifest.

## Held-out study JSON

```json
{"format":"activa-m2-study-v1","training_ids":[],"validation":[{"video_id":"EXPLICIT_ID","path":"/actual/reviewed/video.mp4","source_times_npz":"/verified/original-frame-times.npz"}]}
```

If public category strings differ from the approved classifier vocabulary, add
an explicit category_map and category_map_reviewed_by. No mapping is guessed.
Use scripts/evaluate_perception.py --help for required inputs. It evaluates four
baseline modes and three pooling methods on reviewed validation videos and writes
sample counts,category matrices and baseline macro rankingF1. No optimization or
training is launched. Source-level split review and official scorer parity remain
separate acceptance gates.

The ranking-F1 proxy selects top-ceil(0.15N) samples from each human/prediction
vector, resolving ties by earlier sample. It is not time-weighted or original-
frame F1 and does not enforce duration budgets. Constant tied vectors can match
arbitrary tie choices; that is not evidence of useful predictions. Do not confuse
this score with M3 final highlightF1 or Samsung judge scores.

## Development cache v1

write_cache generates clip/*.npz(times,feats,model_name),importance/*.npz(times,
values),manifests/*.json. Keys include sanitized video ID,input SHA prefix and
configuration/model SHA prefix. The manifest records duration,full input hash,
checkpoint identity,dimension,targetFPS,all array shapes,configuration and payload
hashes. Files are atomic; raw model/video assets remain ignored.

read_cache validates bounded paths,NPZ hashes,identity,shape,float32unitnorm,
exact importance grid/range and original video hash if locally present. Use
allow_pickle=False. This is a development consumer helper only. M3 acceptance
of the format is pending. Optional motion-only CLI cache remains a separately
labeled legacy diagnostic and is not a genuine CLIP handover artifact.

## Evaluation evidence completeness

Current reports also record dataset version/input hashes, exact safe IDs, scorer
version/protocol, macro precision/recall/F1 over identical videos, baseline and
pooling runtime, prompt-template hash and the reproduction command. A positive
full toy-orchestration test exercises this path without protected data. Its purpose
is explicitly TOY-INFRASTRUCTURE-TEST-NOT-QUALITY; no toy scores count as model
quality or grant training clearance.
