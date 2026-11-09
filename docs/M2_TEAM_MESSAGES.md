# M2 progress comments for the existing issues

These are prepared updates, not posting confirmations. Use existing Issues #1,
#2 and #3; do not create duplicates. Last inspected: all OPEN, zero comments and
no linked PRs. API posting is blocked until the saved network rule is applied.

## Issue #1 — M1

```text
[FROM: M2] [TO: M1] [TYPE: REQUEST] [TASK: #1]

New evidence: I inspected m1-actions@fccad75 against canonical main@cc3d61d.
Your actual grid conversion and finalize helpers round-trip and produce an
Action accepted by M4's sanitizer. The controlled fixture preserves seconds,
valid duration boundaries and internal confidence. Your public detect_actions
still returns []; this is not claimed as raw-video model alignment.

M2 now passes 78 standalone tests; the actual M1/M2/M3/M4 source rehearsal passes
237 tests. Real M2 ran through M4 main.run/run.sh without fallback using M4's
explicit TEST vocabulary. Shared decoder/config/contracts were not modified.

Needed: real MP4 action predictions/cache with input hash and timestamp metadata;
confirm resampling/PTS origin, rotation ownership, confidence/vocabulary rules,
Torch/CUDA dependencies and measured runtime after your raw-feature gate.
The M2 decoder remains actual PTS minus stream start (or first PTS), seconds,
with target 2 fps and display-oriented RGB.

Artifact: docs/M2_PATCH_TRANSFER.md, scripts/verify_team_interfaces.py and
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/evidence/M2_team_validation_2026-11-09.json

Acceptance: compare actual M1 predictions and M2 data on one source video and
explicitly acknowledge timeline/runtime compatibility. No ACK is assumed from
the draft reply in your documentation.
Status: OPEN
```

## Issue #2 — M3

```text
[FROM: M2] [TO: M3] [TYPE: REQUEST] [TASK: #2]

New evidence: current m3-selector@cc6f36a consumed genuine 20x512 M2 features
through canonical M4 classes, with empty actions, exact importance grid and
valid local 15% budget. All 237 combined tests pass, including 78 M2 cases.

Controlled early/late/uniform importance curves produced identical clip
locations: [4.2500000005, 5.7499999995] for the 10-second smoke input.
code/optimizer.py:39-54 is explicitly uniform and exposes no importance-only
configuration. This is expected for U, not evidence that B0/B2 exists. With no
action term, please provide/test the intended B0 importance-only mode, then B2.

The fixture generator exports real features, all three curves, input/model
hashes and observed/expected behavior. No features or videos are committed;
recreate them using the documented hash-checked smoke setup and integration
recipe. Production must never read these development artifacts.

Checkpoint identity is unchanged:
ViT-B-32/sha256:9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b
It is the explicit LAION400M_e32 candidate, not OpenAI weights; team freeze
approval remains pending. Grid: actual increasing seconds at target 2 fps;
float32 unit vectors; importance finite [0,1] on exactly those timestamps.

Needed: B0/B2 implementation and acceptance tests; cache-v1/reader acknowledgement;
normalization/model/grid agreement; original-frame/tie/budget F1 conventions and
real-prediction ablation plan. Your current sample-grid scorer is a local proxy.

Artifacts:
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/M2_PATCH_TRANSFER.md
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/scripts/verify_team_interfaces.py
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/evidence/M2_team_validation_2026-11-09.json

Acceptance: genuine cache loads, the intended importance-only objective responds
to controlled peaks, and scorer/safe data rules are agreed with M4. No held-out
F1 or safe FPVSum split is claimed.
Status: OPEN
```

## Issue #3 — M4

```text
[FROM: M2] [TO: M4] [TYPE: REVIEW] [TASK: #3]

New evidence: I tested your canonical main@cc3d61d with actual M1/M2/M3 source.
78 M2 tests and 237 combined tests pass with real pretrained weights. Actual
main.run/run.sh emitted locally schema-valid JSON with no stage fallback and
empty stdout. This used your explicit TEST vocabulary, not official classes.
M1 and M3 remain their documented action/selector stubs.

Fixed the reported M2 vocabulary mismatch. A present vocab module with a missing
configured kit file can use explicit development labels; strict mode, malformed
installed files and unrelated errors still fail. Shared config/contracts/vocab/
video_io/main/schema/run.sh were preserved byte for byte. The platform-neutral
motion test invokes Python directly; Windows itself is still unverified.

Your integration plan proposes --allow-unrelated-histories, but the user's
explicit instruction forbids it. I prepared a source-only M2 patch checked
against your exact main base instead. It preserves your canonical files/README
and proposes only M2 modules/tests/tools and the dependency union for review.
No main push, forced merge, history rewrite, self-merge or premature closure.

Needed: approve the patch-transfer review branch/dependency union, explicit
LAION400M_e32 checkpoint and asset distribution; provide the actual student kit,
supported category data, evaluation manifest, validator and metric/runtime rules.
The canonical source files are now available and are no longer requested as
missing. Data clearance, held-out quality and actual official validation remain
blocked; no TCN training or protected evaluation tuning occurred.

Artifacts:
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/M2_PATCH_TRANSFER.md
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/M2_INTEGRATION_STATUS.md
https://github.com/harshjha15335/ACTIVA/blob/m2-perception/docs/evidence/M2_team_validation_2026-11-09.json

Acceptance: approve the safe related-base transfer and checkpoint, supply actual
official artifacts, then validate the integrated unseen MP4 offline with the
real validator. Local fixture JSON does not satisfy that acceptance condition.
Status: OPEN
```
