# PLmorph sandbox

Phase 1 of the "exciton as a ruler" project: invert a **transition-energy distribution p(E)** into **morphology distribution(s)** on synthetic data.

Start here:
- `PLAN.md`: goals, ideas, figure catalogue (discussion document)
- `SPEC.md`: model and interfaces (the source of truth for code)
- `TASKS.md`: milestones M0–M8 with acceptance criteria
- `AGENTS.md`: rules for the coding agent
- `DECISIONS.md`: dated decisions (confirmed and provisional)
- `RESULTS.md`: progress log, one entry per milestone
- `figures/GALLERY.md`: current figures at a glance

```bash
pip install -e ".[dev]"
pytest tests/acceptance -k m0
```
