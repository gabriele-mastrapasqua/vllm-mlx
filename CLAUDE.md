# CLAUDE.md

## Git workflow

**Always work on a feature branch.** Never commit directly to `main`.

Before starting any task (feature, fix, refactor, experiment):

```bash
git checkout -b feat/<short-description>   # or fix/, refactor/, etc.
```

Commit early and often on the branch. Only merge to `main` via PR after testing.

This prevents half-done or experimental changes from polluting `main` and makes it easy to discard work that doesn't pan out.

## Testing

Run the test suite before considering work done:

```bash
uv run pytest tests/ -v
```

## Project

- Python project managed with `uv`
- MLX-based vLLM-compatible inference server
- Entry point: `vllm_mlx/server.py`
- Tool call parsers: `vllm_mlx/tool_parsers/`
