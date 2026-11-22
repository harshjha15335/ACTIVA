# AI disclosure

Team LensLog used AI coding assistants to build ACTIVA. This page says which, for what,
and how we checked their work.

## At inference time: no generative AI

`./run.sh` runs three pretrained perception models, all third-party and credited in the
README. We trained none of them:

- **CLIP ViT-B/32** (OpenCLIP, LAION-400M): video label and frame importance.
- **SlowFast** (EPIC-KITCHENS-100 checkpoint): action features.
- **ActionFormer** (EPIC-KITCHENS-100 checkpoints): verb and noun detection.

No large language model, chatbot or online AI service is called, and `run.sh` makes no
network requests. The highlight selector is our own deterministic code.

## During development: AI coding assistants

- **Tools:** Claude Code (Anthropic) and Codex (OpenAI), used as coding agents by the team members.
- **What they did:**
  - wrote and edited code, tests and documentation from our instructions;
  - ran experiments and test suites;
  - drafted board messages and reports;
  - built the presentation and edited the demo video from the team's content and recorded voices;
  - ran independent review passes ("judge", planner and edge-case reviews in `tmp/review/`, not committed) that found issues we then fixed, for example 4.00 s highlights that were 3.9999 s in floating point (D22).
- **How we set them up:** in the open, in this repo.
  - `AGENTS.md` and `CLAUDE.md` give the agents' rules.
  - `docs/agents/M1.md` to `M4.md` give each member's scope.
  - `docs/agent_board_protocol.md` says how the agents coordinated through GitHub issues.
- **Where it shows in the history:** commits written with Claude Code carry a `Co-Authored-By: Claude` trailer, and PRs opened with it end with "Generated with Claude Code". We kept these on purpose.

## Human oversight

- **Work was done on role branches and merged into `main`, mostly through pull requests.** CI runs the full test suite (450+ tests). Merges were made from a team member's account, some by a coding agent under that member's standing instruction once the checks passed.
- **Design decisions are made by the team and recorded** in `docs/decisions.md`, with the evidence behind them.
- **Reported numbers come from scripts in this repo** and were re-measured before we wrote them down: validator results, run times, ablations and label accuracy. Experiments followed rules fixed before the results were seen, and were run once on held-out data.
- **We avoided leakage:** hidden-set videos and annotations were never used to tune anything (`docs/leakage.md`).
- **Each member owns one pipeline stage** and can explain it without the agents:
  - M1: actions;
  - M2: perception and labels;
  - M3: highlight selection;
  - M4: integration.

## What the AI did not do

- It did not have the final say on the approach or the trade-offs. The team made the final call, recorded in `docs/decisions.md`.
- It did not train or fine-tune any model.
- It did not produce or edit ground-truth labels, and no annotations of the evaluation videos were used to tune anything.
- It did not generate the output JSONs by hand: every file in `outputs/` is written by `./run.sh`.
