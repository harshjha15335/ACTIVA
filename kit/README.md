# Official student kit (local only)

Unzip `Student package_track04 (1).zip` (SHA256
`ddf7ee8d2b8b9fb1363c1e72ae95da9144f64573cbca5a1b7a2378f41bbd0c09`; ask M4 or M2
for it) and copy its files here:

- `supported_classes.json`: read by `code/vocab.py` (override with `ACTIVA_VOCAB`)
- `manifest.csv`: read by `scripts/run_manifest.py` and `scripts/validate_outputs.py`
- `TEAM_README.md`, `P02_109.json`, `skiing_03.json`

Everything here except this README is git-ignored until the organisers confirm
the kit may be published (`docs/kit_checklist.md` #9). Tests that need the real
kit skip when it is absent (CI never has it).
