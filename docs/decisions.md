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
