# FPVSum leakage controls — current status (18 November 2026)

**No supervised FPVSum training or manifest-based model/prompt tuning is
authorised.** The official Samsung evaluation manifest is now available to the
team privately; it must not be committed to this public repository. Public
metadata and source-ID differences alone do not establish a safe development
split. No model-quality or submission-score improvement is claimed here.

## Data provenance and candidate separation

The pinned upstream FPVSum inventory is
`github.com/azuxmioy/fpvsum` at
`9af31f9027ef2b3f5f34a39bf17df5b120a9cc25`.
The inspected inventory has **98 videos in 14 classes** (56 labelled,
42 unlabelled). The team's private Samsung manifest includes 49 FPVSum
videos. An initial 49 different-YouTube-ID candidate set was reduced by
three BikePolo `_2` continuation recordings whose `_1` portions overlap
manifest recordings, leaving **46 source-ID/stem-screened candidates**.
Of these, 34 unlabelled videos in 13 sports classes were locally acquired
and feature-cached. **Human visual review of aliases, re-encodes, edits and
shared source-content groups remains NOT DONE**. Accordingly these 34
are candidates, not approved independent evaluation videos.

The 34-video baseline from Aarush's genuine release-checkpoint CLIP
caches is **28/34 = 82.4% category accuracy** (inventory category labels);
the private cache ZIP's SHA-256 and 34 cache/receipt records were checked
by M2 and acknowledged in #33. The bundle is not in the repository.
This is **preliminary baseline-only evidence**, not manifest performance,
not official judging results, and not clearance to run prompt variants.
All seven independently labelled non-manifest FPVSum candidates were
unavailable via the documented public sources, so the required independent
highlight-F1 no-harm study cannot currently run. The excluded BikePolo
continuations must never be counted as clean candidates.

## Freeze-week decisions (D-a / D-b / D-c)

- **D-a, manifest annotations: report only.** After the 21 November
  frozen outputs have been committed and hashed, any permitted official
  manifest-annotation mAP check is descriptive only. It must never guide
  prompt, mask, hyperparameter or model changes.
- **D-b, title-card mask: NO-GO.** M3's independent test of PR #29 on
  10 EPIC development videos found three sources above the predeclared
  1% masked-frame bound (7.7645%, 2.3631%, 1.2220%); the
  0.5-percentage-point stability bound also failed at three of four
  action densities. Keep the candidate disabled, do not merge PR #29,
  and do not retune from its failures. A proposed sport-only extension
  in #33 likewise remains unshipped without visual false-positive review
  and independent labelled-video highlight-F1 no-harm evidence.
- **D-c, independent FPVSum acquisition:** Public videos/caches are kept
  in ignored local paths. Filter by manifest video/source identifiers,
  known continuation groups, hashes and human-reviewed content
  similarity. Source-ID exclusion and a 34-video cache receipt are
  **not** final independence approval. In #33, M2 requested a human
  reviewer, method, manifest/source hash binding and specific
  exclusions before freezing an eligible set or evaluating any prompt
  variants. If approved, at most three *predeclared* variants may be
  evaluated once; adoption also requires 10/10 EPIC dev videos still
  classified as `cooking`.

**Current status:** no reviewed-safe FPVSum training set; no cleared
independent labelled highlight-F1 set; prompt variants **NOT RUN**;
title-card mask **OFF / NO-GO**. Production does not consume public
video manifests, human annotations, or development caches.

## Historical integrity and unresolved upstream issues

The upstream parsed files include 56 processed-score records and
978 raw annotator records. `Jumping_mU-p2wuk90` versus
`Jumping__mU-p2wuk90`, and raw `BikePolo` versus CSV
`Bikepolo`, remain identity/category discrepancies requiring
explicit evidence-backed reviewed aliases; case/underscore similarities
are not proof of identity. Of the raw annotation records, 160/978
contain finite fractional values in [0,1]; they are preserved instead
of coerced to binary. The published processed Score is not shown to
be a naive mean of raw Label annotations.

Processed-score frames are not supplied with verified decoded original
PTS timestamps. Annotation resampling requires reviewed source video,
original aligned timestamps, duration and a checked identity mapping;
nominal downloaded FPS is insufficient. The upstream repo has no
explicit dataset LICENSE file: public access does not grant reuse
permission.

## Review gates and tooling

1. Bind every candidate to the exact **private official-manifest hash**,
   source content group, local input SHA-256, reviewer identity, review
   date and a factual alias/content-comparison receipt.
2. Exclude every known overlap and all unresolved candidates, including
   alternate uploads, continuations, trims or re-encodes. The absence
   of a matching identifier is not a clearance decision.
3. Maintain held-out separation by recording/source group, not individual
   frames. Do not access manifest annotations when selecting prompts.
4. Keep data, private kit, review media, caches and annotations in ignored
   local locations. Only public-safe aggregate receipts and source
   policies belong in GitHub.

M2 tools: `fpvsum_data.py` enforces reviewed split/hash decisions;
`fingerprint_videos.py` provides candidate fingerprints only;
`audit_ids.py` reports preliminary ID/hash overlaps;
`evaluate_perception.py` fails without reviewed source decisions.
Its sample-grid ranking-F1 is a development proxy, **not** Samsung
final highlight-F1. See `docs/evaluation_formats.md` for local adapter
formats and `docs/evidence/` for explicitly qualified receipts.
