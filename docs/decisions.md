# Decisions

One entry per consequential technical decision (guide A6). Newest last.
Members append their own sections when their branches land.

## M4 integration

**D1 (2026-11-09) Fresh `main`.** The three role branches share no git history,
so `main` starts from `m4-integration` (its own root commit). Each owner merges
`main` into their branch with `--allow-unrelated-histories`, then opens a PR.
See `docs/integration_plan.md`.

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
4 invalid video, 5 configuration. Output JSON is written atomically.

**D9 No action `score` in the JSON** until the kit allows it
(`config.INCLUDE_ACTION_SCORE`).


## M2 decisions proposed for review

# M2 implementation decisions — pending shared-owner acceptance

These document M2-owned remediation; they do not amend shared contracts or claim
M4 approval. Shared config.py,contracts.py,video_io.py and requirements.txt are unchanged.

1. Configuration precedence is explicitCLI > environment > shareddefaults.
   Validate through the M2 perception_options adapter; keep M4's config API intact.
2. Keep original OpenAI as its existing default. Validate the explicitly selected
   OpenCLIP LAION400M_e32 QuickGELU B32 alternative through the already permitted
   official GitHub release. Both have512 dimensions, but weights differ. Team
   approval/evidence-based final selection remains pending.
3. Record published checksum-prefix verification honestly, plus full local source
   and converted hashes. Never relabel the observed full hash as an independently
   published full digest. No verification or TLS was disabled.
4. Use actual checkpoint content identity in ClipFeats.model_name. Shared dataclass
   shapes/signatures remain unchanged; the cache manifest is M2 development data.
5. Canonical M4 vocab.load() is preferred and dev-label conflict fails. Missing
   canonical vocabulary can be made fatal through strict mode. Fixture labels do
   not certify official compatibility.
6. Development caches use atomic NPZ/JSON, input/config/checkpoint identity and
   strict read validation. M3 must accept the v1 format; no acceptance is assumed.
7. FPVSum processed Score is preserved; raw Label can be fractional. Do not infer
   raw-to-processed averaging, an underscore repair or case normalization. Explicit
   evidence-backed aliases are required. Original frame PTS must be supplied.
8. ID/hash/content fingerprints can identify review candidates, never establish
   safety by lack of a match. Training/validation require a reviewed manifest-bound
   decision and disjoint source groups/hashes. No FPVSum run is cleared here.
9. M2 evaluation uses a labeled sample-grid ranking-F1 development proxy. It is
   not original-frame or final-highlight F1 and does not enforce interval budgets.
   M3/M4 scorer parity remains necessary before official comparisons.
10. TCN remains untrained and outside production predict_importance. Do not launch
    training before genuine features, annotation alignment, safe splits, baseline
    evaluation and approved scorer conventions are established.
11. Preserve M1/M3/M4 files and unrelated Git histories. Shared foundation migration,
    merges and submission require their owners/humans. Messages are drafts only.

## M2 canonical integration decisions — 9 November 2026

- Canonical source is main@cc3d61d; M1@fccad75 and M3@cc6f36a are available.
  Tests use immutable source copies, not edits to those owner branches.
- User explicitly forbids unrelated-history merging. Despite M4's proposed merge
  command, prepare a source-only patch checked against canonical main instead.
- Missing installed vocabulary modules and missing kit data are distinct states.
  Only a declared M4 VocabError plus a demonstrably absent configured file may
  use explicit development labels; strict mode and malformed files still fail.
- Real M2 passes through the actual M4 runner/schema with fixture labels, not
  official vocabulary. This is canonical API evidence, not official compliance.
- Existing issues #1/#2/#3 stay OPEN; no owner ACK was observed. API egress is
  blocked and its domain addition was saved; no comment is considered sent until
  a successful GitHub response is recorded.
