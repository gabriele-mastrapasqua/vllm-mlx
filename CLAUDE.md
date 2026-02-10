# CLAUDE.md

## Git workflow

**Prefer working on a feature branch** for non-trivial changes.

```bash
git checkout -b feat/<short-description>   # or fix/, refactor/, etc.
```

Commit early and often. Direct merge to `main` and push is allowed — no PR required.

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
