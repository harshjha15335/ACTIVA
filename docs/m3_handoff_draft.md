# M3 handoff drafts - 11 November 2026

Prepared locally; not posted or sent.

```text
[FROM: M3] [TO: M2, M4] [TYPE: HANDOFF] [TASK: #2 / M3-ACTION-AWARE-HIGHLIGHT-OPTIMIZER]
Context: Uniform production stub replaced by joint predicted-action coverage
and integrated importance. U/I/A/J use the canonical four-argument interface.
Artifact: m3-action-aware-selector; code/candidates.py, code/optimizer.py,
eval/metrics.py, scripts/run_ablation.py, owned tests and M3 docs.
Acceptance evidence: controlled importance AND action relocation, duplicate
invariance, canonical minimum/rounding checks, synthetic U/I/A/J and the existing
check_m3_handoff.py --require-importance-sensitive on a SYNTHETIC v1 cache.
Cache/grid: accept activa-m2-cache-v1 via read_cache for development only:
hash-identified normalized float32 features, matching actual PTS, importance
[0,1]. Production never reads caches. CLIP vectors are not directly scored;
M2 importance supplies novelty and distinct pairs supply diversity.
Local metrics: explicit coverage overlap, annotation-grid F1, ceil(.15*N),
earliest ties, half-open intervals; official scorer parity remains unverified.
Needed: genuine M2 cache from >=26.67s with source/model hashes and reviewed
safe labeled data for quality ablations. The original 10s cache necessarily
violates 15% when a >=4s clip is emitted. See M3_SELECTOR.md/results.md.
Status: OPEN (implementation prepared; real/held-out evidence pending)
```

```text
[FROM: M3] [TO: M1] [TYPE: REQUEST] [TASK: M3-real-action-ablation]
Needed: genuine raw-MP4 predictions on an approved development video: source and
checkpoint SHA-256, exact timestamp origin, verb/noun, start/end, confidence,
preprocessing identity/runtime. Prefer the same source as M2's cache.
Acceptance: hashes match, actions pass canonical bounds, predicted and separately
labeled oracle selection experiments reproduce. Published-feature evidence is
separate from raw-MP4 outputs.
Status: OPEN
```

```text
[FROM: M3] [TO: M4] [TYPE: BLOCKER] [TASK: official-metric-parity]
Context: authentic ZIP verified; promised validate.py/scorer absent.
Needed: organizer coverage overlap, F1 target/frame/tie rules, soft budget
penalty and short-video clarification. Existing D13 single-4s policy is preserved
and logged, never labeled strict-budget compliant. No shared files were edited.
Acceptance: authoritative scorer/rules supplied and checked on agreed cases.
Status: OPEN
```
