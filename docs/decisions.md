# Decisions

One entry per consequential technical decision (guide A6). Newest last.
Members append their own sections when their branches land.

## M4 integration

**D1 (2026-11-09) Fresh `main`.** The three role branches share no git history,
so `main` starts from `m4-integration` (its own root commit). Members' files
are copied onto landing branches cut from `main` (PRs #5-#7), not merged with
`--allow-unrelated-histories`; role branches are left untouched. M2 asked for
this route in #3. See `docs/integration_plan.md`.

**D2 Contracts.** `code/contracts.py` is guide A5 field-for-field. It matches
the copy M2 bootstrapped on `m2-perception`, with comments added.

**D3 video_io adopted from M2.** Same decoding and timestamps as
`m2-perception`: time = PTS minus stream start (or first PTS); the yielded
frame is the first one at or after each 1/f slot, carrying its real PTS
(VFR-safe; the guide's "closest frame" wording is replaced by this rule, error
is under one source frame). Changes: errors are `VideoError` (a `ValueError`),
duration falls back to the container, and the legacy clockwise `rotate` tag is
converted to the display-matrix convention. `VideoInfo.rotation` is
counter-clockwise degrees in {0, 90, 180, 270}; frames come out display-oriented.

**D4 config.py keeps M2's `Settings`/`settings()` verbatim**, because
`perception_options.py` imports them. Shared constants sit above them.

**D5 vocab.load()** returns a 4-tuple NamedTuple `(verbs, nouns, categories,
is_valid_pair)`, the shape M2's adapter checks. The JSON layout it parses is
an assumption until the kit arrives (`docs/kit_checklist.md` #1). A missing
vocabulary is a configuration error: exit code 5, no JSON written.

**D6 Rounding never breaks validity.** Highlights round inward (start up, end
down), ends cap at the rounded-down duration, zero-length intervals are dropped.
`OUTPUT_DECIMALS = 2` until the kit says otherwise.

**D7 Stage guards.** `main.py` validates every module's output against the
contracts (types, grids, ranges, budget). A crash or invalid output means a
logged traceback and a safe fallback. Actions are filtered one by one (bad
vocabulary or non-finite values dropped, times clamped into the video) rather
than discarding the whole list.

**D8 Exit codes.** 0 ok, 1 internal error, 2 usage, 3 input not found,
4 invalid video, 5 configuration, 6 a stage failed with `ACTIVA_STRICT=1` (no JSON
written). Output JSON is written atomically.

**D9 No action `score` in the JSON** until the kit allows it
(`config.INCLUDE_ACTION_SCORE`).


**D10 (2026-11-11) Output format from the student kit.** `schema.py` now writes
the kit's 6 fields (team_id, video_id, duration_sec, label {name, score},
actions [{segment, verb, noun, score}], highlights [{segment}]) and supersedes
the 4-key layout in D5/D6/D9. Internal contracts (`contracts.py`) are unchanged;
conversion happens only at the output boundary. Action `score` is always written.

**D11 Vocabulary layout.** `vocab.py` reads `labels`/`verbs`/`nouns`; the kit has
no pair list, so every verb x noun is valid. `Vocab.categories` holds the labels.

**D12 Highlights >= 4 s.** Clips under 4 s are dropped before checks (the
scorer ignores them); touching clips are merged. The fallback uses 6 s clips.

**D13 Short videos (provisional).** When 15% of a video is under 4 s, we output
exactly one 4.0 s clip: a file without a >= 4 s segment is invalid (scores 0),
while the budget is a warning with a soft penalty. `config.SHORT_VIDEO_SINGLE_CLIP`;
pending organiser confirmation (`docs/kit_compliance.md` Q2).

**D14 team_id** comes from `config.TEAM_ID` or `ACTIVA_TEAM_ID`; missing is a
configuration error (exit 5). No placeholder ID is ever written.
2026-11-11: no official team ID was issued, so `TEAM_ID = "LensLog"`, the
registered team name (provisional; organisers asked, `docs/kit_compliance.md` Q9).

**D15 Strict mode.** `ACTIVA_STRICT=1` makes any stage failure exit 6 with no
JSON. Default stays fallback-on for submission runs, because a missing file
scores 0; strict mode is for release checks and the batch runner's verification.

**D16 Local spec validator.** `scripts/validate_outputs.py` implements our
reading of the README while the official `validate.py` is missing. It is not the
official validator; results are labelled that way.

**D17 Default CLIP weights.** `config.settings().weights` defaults to
`weights/ViT-B-32-laion400m_e32.safetensors`, the file M2's
`scripts/prepare_weights.py --preset laion400m_e32` writes and validated. The old
default (`weights/ViT-B-32.safetensors`) was never produced, so a plain
`./run.sh` silently fell back without CLIP. `ACTIVA_CLIP_WEIGHTS` still overrides.

**D18 (2026-11-13) Skip damaged packets.** `video_io.iter_frames` skips a packet
the decoder rejects and logs the count, as ffmpeg does, instead of failing the
clip stage. Found in the full run: P29_05 and P30_08 have such packets, and their
MD5s match EPIC's official list, so the organisers' copies are the same. With the
fix, all 2,245 CLIP frames of P30_08 decode (16 packets skipped, no gap over
0.5 s). A file with no decodable frames is still an error.

**D19 (2026-11-15) Submission packaging.** A clean clone must run `./run.sh`:
- **Model files:** the ~1.75 GB of weights are git-ignored and too large for Git
  (GitHub's 100 MB limit). They live in the repo's `weights-v1` release, as unchanged
  mirrors of the upstream files, and `scripts/setup_weights.py` installs them with
  every SHA-256 checked. `--upstream` rebuilds them from the original authors.
- **Missing models are an error:** `main.py` exits 5 with a pointer to
  `setup_weights.py`, instead of quietly writing fallback outputs.
  `ACTIVA_REQUIRE_MODELS=0` restores the old behaviour; the test suite uses it, since
  it has no weights.
- **Device:** `ACTIVA_DEVICE` defaults to `auto`, so CUDA is used when available. The
  action stage is about 12x slower on CPU. *(Measured 17 Nov on P26_41: actions 12x,
  whole pipeline 9x slower on an i7-14650HX; numbers in the README.)*
- **Class list:** `kit/supported_classes.json` stays local, like the rest of the kit,
  until the organisers answer Q7 (team vote, 15 Nov). The README tells whoever runs
  `run.sh` to copy it from their kit first; without it, `run.sh` exits 5 with that
  hint. The organisers re-running `run.sh` have the kit. *(Superseded by D20.)*

**D20 (2026-11-17) `run.sh` works without the kit.** An independent review pointed out
that the kit never says the organisers' re-run environment has `kit/`, so relying on it
risks every output. The kit file still stays local (D19's vote), but `vocab.load()` no
longer needs it:
- If `ACTIVA_VOCAB` is unset and `kit/supported_classes.json` is absent, the vocabulary
  is rebuilt the way the kit's own notes describe it: verbs and nouns are the `key`
  column of the EPIC-100 class CSVs the action engine already installs, in file order,
  plus `other`; labels are the 14 FPVSum categories plus `cooking`, sorted, then `other`
  (`config.FPVSUM_CATEGORIES`). A warning is logged.
- `tests/test_vocab_fallback.py` checks the rebuilt lists equal the kit file exactly,
  order included (local only, since CI has neither). A kit file that is present always
  wins; an explicit `ACTIVA_VOCAB` that doesn't exist, or a malformed file, is still exit 5.
- Measured 17 Nov on non-manifest P30_06 with `ACTIVA_STRICT=1`: exit 0 with and
  without `kit/`, byte-identical JSON.

**D21 (2026-11-17) No kitchen actions on sports videos.** The action model knows only
EPIC-KITCHENS verbs and nouns, so on FPVSum videos it produced ~2,000 low-score kitchen
actions each ("eat light" on a ski run); the kit allows `actions: []` for FPVSum.
- `main.py` now runs clip → importance → category → actions → highlights (no stage
  used another's output out of order, so nothing else changes), and skips the action
  stage when the category stage succeeded **and** its label is in
  `config.FPVSUM_CATEGORIES`. `cooking`, `other` and a category fallback run actions as
  before, so an uncertain label never costs a kitchen video its actions.
  `ACTIVA_ACTION_GATE=0` restores the old behaviour.
- Robust to either reading of the action metrics: no effect if they are EPIC-only (as
  the kit says), fewer false positives if every file is pooled.
- Risk: an EPIC video labelled as a sport would lose its actions. Exposure: 138/138
  manifest EPIC outputs and 10/10 non-manifest dev videos are labelled `cooking`.
- Measured 17 Nov (`ACTIVA_STRICT=1`): non-manifest P30_06 and P02_04 byte-identical
  to `main`. FPVSum `Ski_-Z5a4UQmZLs` (a validity and timing check only): same label and
  highlights as the 14 Nov run, 0 actions instead of 2,000, 17.4 s instead of a full
  action pass. FPVSum highlights change only where an action scored >= 0.2 (2/49 videos).

**D22 (2026-11-17) 4.00 s clips pass a literal float check.** On the 0.01 s grid a
4.00 s clip is often 3.9999999999999964 s as floats (`32.19 - 28.19`). Our validator
allowed 1e-6 slack, so it never flagged this; a validator testing `end - start >= 4.0`
would. In the 14 Nov outputs, 35 clips in 33 files were float-short, and in 3 files
(P02_14, P26_37, P28_23) every clip was, which would fail "at least one segment
>= 4 s" and score 0. `schema.to_output` now widens such a clip by 0.01 s at its end,
or else its start, staying inside the video and clear of its neighbours, and
`validate_output` checks 4 s literally. Applied to the 187 outputs: all 35 fixed, no
file over 15%. Found by the edge-case review; the 21 Nov run picks it up.

**D23 (2026-11-17) CUDA out-of-memory retries on the CPU.** On a small or shared GPU
the action stage ran out of memory, the category stage then failed too, and the run
wrote 0 actions and label `other` with exit 0, which a re-run can't tell from a real
result. `main._stage` now retries a stage once on the CPU after a CUDA out-of-memory
error (`ACTIVA_DEVICE=cpu` for the rest of the run, the action engine reloaded). Other
errors, a second failure and strict mode behave as before. Measured 17 Nov on a 40 s
test clip with the GPU capped at 1.2 GB: before, 0 actions and `other`; after, exit 0
and a JSON identical to a pure-CPU run (1,849 actions, same label and highlights).

## M2 perception decisions — current as of 18 November 2026

This section supersedes the pre-integration process notes archived with the
`docs/M2_*.md` handoff reports. M2 changes do not modify M1/M3/M4-owned
contracts or override decisions D17–D23 above.

- **CLIP and category baseline.** Production uses the release
  `ViT-B-32-laion400m_e32.safetensors` model, 512-dimensional normalized
  features on the actual 2-fps PTS grid, novelty importance, and mean-pooled
  zero-shot category scores. Development cache v1 is for offline analysis only;
  the inference path must never load these caches.
- **Kit separation.** A local kit vocabulary takes precedence when supplied.
  D20 defines the public no-kit fallback. Training and evaluation material,
  official manifest rows, video media and private caches must not be published.
- **Leakage protection (D-a/D-c).** The official manifest is now available on
  the team's private machines. Its video/source identities must be excluded from
  development evaluation; a different YouTube ID alone does not prove content
  separation. Human-reviewed alias/content checks and source grouping are
  required before declaring the remaining FPVSum candidates safe. Manifest
  annotations may be used for report-only analysis after frozen outputs are
  hashed (D-a), never for prompt/model tuning.
- **Rejected title-card mask (D-b).** Draft PR #29 failed its predeclared
  10-video EPIC false-positive and M3 stability gates (#27/#28, PR #32).
  Decision: **NO-GO**; keep `ACTIVA_CARD_MASK=0` and do not merge/activate
  that candidate. A sport-only candidate proposed in #33 is separately blocked
  by unavailable independent labelled FPVSum videos and missing highlight-F1
  evidence. No post-result threshold tuning.
- **Category-prompt study (#33).** 34 non-manifest source-ID/stem-screened,
  unlabelled FPVSum recordings have genuine CLIP caches and a preliminary
  baseline of 28/34 correct (82.4%). Human source/content-disjointness
  clearance has NOT been completed. No prompt variant has yet been evaluated.
  Once cleared, define at most three prompt variants before one-pass
  scoring; acceptance requires at least three net additional correct
  classifications on the fixed eligible set and 10/10 EPIC dev examples still
  labelled `cooking`.
- **Remaining verification.** Offline clean-room checks on a fresh clone,
  without `kit/`, network access during `run.sh`, or private artifacts in the
  repo, are scheduled for 19 and 21/22 November (#27). No unrun local test or
  human review is treated as completed.

Historical M2 development notes are preserved in `docs/archive/`. See
`docs/leakage.md`, #27, #33 and D20–D23 for current release gates.

## M3 selector decisions - 11 November 2026

- Replace the uniform stub with bounded greedy nonoverlapping interval packing.
  Distinct pair credit is max per-instance confidence times thresholded union
  coverage; repeated copies cannot accumulate credit. Importance is integrated
  on actual timestamps. U/I/A/J are genuine signal ablations.
- Use M4's output time grid before selection; clips are already >=4 s after
  inward rounding. Preserve M4's existing D13 short-video exception with an
  explicit warning and strict-budget-failure reporting, pending organizers.
- Direct CLIP facility-location scoring/local search are deferred. M2 importance
  supplies novelty and predicted action pairs supply diversity. No measured
  held-out evidence justifies added objective components or frozen settings yet.
- Accept M2 v1 caches only for development experiments through its validating
  reader. Production has no cache/annotation reads. Real quality comparisons
  require a reviewed manifest-bound split and actual M1/M2 artifacts.
- Official metric parity is unresolved. The kit confirms metric weights/datasets
  but lacks its advertised validator/scorer and exact coverage/F1/penalty rules.
  All reported synthetic/local scores remain explicitly labeled proxies.
