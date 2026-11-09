# Instructions for coding agents

ACTIVA turns a 2-10 minute first-person MP4 into one JSON (category label,
every action as verb+noun+start+end, highlights totalling at most 15% of the
video). Entry point: `./run.sh <input.mp4> <output.json>`.

Before writing code:

1. Read `docs/agent_board_protocol.md` (how members' agents coordinate).
2. Read your role file: `docs/agents/M1.md` ... `docs/agents/M4.md`.
3. Code against `code/contracts.py`. Changing it, `config.py` or `vocab.py`
   needs M4's approval on the board first.

Hard rules:

- Stay inside the files your role owns; work on your role branch; PR into `main`.
- Never hardcode video IDs or ground-truth times, and never read `dev_cache/`
  from the inference path.
- Inference is offline: local weights only, no downloads at runtime.
- Never invent official rules, scores or dataset facts. If it must come from
  the student kit, write "VERIFY" and ask (see `docs/kit_checklist.md`).
- Logs go to stderr. Run `python -m pytest -q` before every push.
