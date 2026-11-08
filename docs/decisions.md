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
