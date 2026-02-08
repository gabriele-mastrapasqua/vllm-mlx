# OpenCode + vllm-mlx Setup

## Quick Switch (recommended)

One command to configure OpenCode + start the server:

```bash
# Interactive — pick from a list
./scripts/switch-model.sh

# Direct — pass the model
./scripts/switch-model.sh mlx-community/Qwen2.5-3B-Instruct-4bit
```

This will:
1. Write `~/.config/opencode/opencode.json`
2. Write `~/.local/share/opencode/auth.json`
3. Clear OpenCode cache
4. Start the vllm-mlx server with all params (batching, tool calling, cache, etc.)

Then in another terminal:
```bash
opencode
```

## Manual Setup

### Server

```bash
uv run vllm-mlx serve mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 0.0.0.0 --port 8000 --api-key sk-dummy-prova \
  --continuous-batching \
  --tool-call-parser qwen --enable-auto-tool-choice \
  --max-tokens 32000 --max-num-seqs 8 \
  --prefill-batch-size 64 --completion-batch-size 128 \
  --cache-memory-percent 0.60 --stream-interval 1 \
  --rate-limit 3600 --timeout 900
```

Debug logging:
```bash
# Same command + --log-level debug
# Shows full prompts, tool injection, cache details
```

### OpenCode Auth (`~/.local/share/opencode/auth.json`)

```json
{"vllm-local": {"type": "api", "key": "sk-dummy-prova"}}
```

### OpenCode Config (`~/.config/opencode/opencode.json`)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "vllm-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vllm-mlx Local",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-dummy-prova"
      },
      "models": {
        "default": {
          "name": "Local Model",
          "tools": true,
          "toolChoice": "auto"
        }
      }
    }
  },
  "model": "vllm-local/default",
  "small_model": "vllm-local/default",
  "agent": {
    "coder":      { "model": "vllm-local/default" },
    "summarizer": { "model": "vllm-local/default" },
    "task":       { "model": "vllm-local/default" },
    "title":      { "model": "vllm-local/default" }
  },
  "mcp": {
    "local-mcp": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-everything"],
      "enabled": true
    }
  }
}
```

## Supported Models

| Model | Parser | Notes |
|---|---|---|
| Qwen2.5 1.5B/3B/7B/14B Instruct 4bit | `qwen` | 7B+ recommended for tool calling |
| Qwen2.5-Coder-7B-Instruct-4bit | `qwen` | Code-focused |
| Qwen3-8B-4bit | `qwen` | |
| Llama-3.2-3B / 3.3-70B Instruct 4bit | `llama` | |
| Mistral-7B-Instruct-v0.3-4bit | `mistral` | |
| Hermes-3-Llama-3.1-8B-4bit | `hermes` | |

## Test Tool Calling

```bash
./tests/test_tool_call_curl.sh http://localhost:8000 sk-dummy-prova
```
