# Experimental Scripts

This directory holds one-off plotting, rerun, and diagnostic scripts.

Run them from the repository root so package imports resolve and outputs
continue to land in the expected project directories:

```bash
PYTHONPATH=. .venv/bin/python scripts/experiments/<script>.py
```

These scripts are not the stable public API. Prefer the package entry points in
`mah/`, `sfr/`, `ssp/`, and `uvlf/` for reusable work.
