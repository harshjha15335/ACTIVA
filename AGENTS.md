# Instructions for coding agents

ACTIVA turns a first-person MP4 (manifest: 11 s to 33 min) into one JSON in the
student kit's format: team_id, video_id, duration_sec, label {name, score},
every action as {segment, verb, noun, score}, and highlights {segment} that are
each >= 4 s and total at most 15% of the video (see `code/schema.py`).
Entry point: `./run.sh <input.mp4> <output.json>`.

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
- Never invent official rules, scores or dataset facts. The student kit is
  local-only (`kit/`, git-ignored; ask M4 for the ZIP). Its rules are summarised
  in `docs/kit_compliance.md`; open questions are in `docs/kit_checklist.md`.
- Logs go to stderr. Run `python -m pytest -q` before every push.
