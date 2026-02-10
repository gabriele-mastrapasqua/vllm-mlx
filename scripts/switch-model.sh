#!/usr/bin/env bash
# switch-model.sh — Switch vllm-mlx model + OpenCode config in one shot.
#
# Usage:
#   ./scripts/switch-model.sh                              # interactive
#   ./scripts/switch-model.sh mlx-community/Qwen2.5-3B-Instruct-4bit  # direct
#
# What it does:
#   1. Picks model + tool parser
#   2. Writes ~/.config/opencode/opencode.json
#   3. Writes ~/.local/share/opencode/auth.json
#   4. Clears OpenCode cache
#   5. Starts the vllm-mlx server

set -euo pipefail

# ── Config ──────────────────────────────────────────────
PORT=8000
HOST="0.0.0.0"

# Server defaults (max_tokens auto-detected from model config)
MAX_NUM_SEQS=8
PREFILL_BATCH_SIZE=64
COMPLETION_BATCH_SIZE=128
CACHE_MEMORY_PERCENT=0.60
STREAM_INTERVAL=1
RATE_LIMIT=3600
TIMEOUT=900

OPENCODE_CONFIG="$HOME/.config/opencode/opencode.json"
OPENCODE_AUTH="$HOME/.local/share/opencode/auth.json"
OPENCODE_CACHE="$HOME/.cache/opencode"

# ── Known models (name -> tool_call_parser) ─────────────
declare -A MODELS
MODELS["mlx-community/Qwen2.5-1.5B-Instruct-4bit"]="qwen"
MODELS["mlx-community/Qwen2.5-3B-Instruct-4bit"]="qwen"
MODELS["mlx-community/Qwen2.5-7B-Instruct-4bit"]="qwen"
MODELS["mlx-community/Qwen2.5-14B-Instruct-4bit"]="qwen"
MODELS["mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"]="qwen"
MODELS["mlx-community/Qwen3-8B-4bit"]="qwen"
MODELS["mlx-community/Llama-3.2-3B-Instruct-4bit"]="llama"
MODELS["mlx-community/Llama-3.3-70B-Instruct-4bit"]="llama"
MODELS["mlx-community/Mistral-7B-Instruct-v0.3-4bit"]="mistral"
MODELS["mlx-community/Mistral-Small-3.1-24B-Instruct-2025-04-01-4bit"]="mistral"
MODELS["mlx-community/Devstral-Small-2505-4bit"]="mistral"
MODELS["mlx-community/Ministral-8B-Instruct-2410-4bit"]="mistral"
MODELS["mlx-community/NousResearch-Hermes-3-Llama-3.1-8B-4bit"]="hermes"

# ── Select model ────────────────────────────────────────
if [ -n "${1:-}" ]; then
  MODEL="$1"
else
  echo "Available models:"
  echo ""
  i=1
  MODEL_KEYS=()
  for key in "${!MODELS[@]}"; do
    MODEL_KEYS+=("$key")
    printf "  %2d) %s  [%s]\n" "$i" "$key" "${MODELS[$key]}"
    ((i++))
  done
  echo ""
  printf "  %2d) Custom model\n" "$i"
  echo ""
  read -rp "Pick a number: " choice

  if [ "$choice" -eq "$i" ] 2>/dev/null; then
    read -rp "Model name (e.g. mlx-community/MyModel-4bit): " MODEL
  elif [ "$choice" -ge 1 ] && [ "$choice" -lt "$i" ] 2>/dev/null; then
    MODEL="${MODEL_KEYS[$((choice-1))]}"
  else
    echo "Invalid choice"
    exit 1
  fi
fi

# ── Resolve parser ──────────────────────────────────────
if [ -n "${MODELS[$MODEL]+x}" ]; then
  PARSER="${MODELS[$MODEL]}"
else
  echo ""
  echo "Unknown model: $MODEL"
  echo "Select tool call parser:"
  echo "  1) qwen"
  echo "  2) llama"
  echo "  3) hermes"
  echo "  4) mistral"
  echo "  5) none (no tool calling)"
  read -rp "Pick [1-5]: " pchoice
  case "$pchoice" in
    1) PARSER="qwen" ;;
    2) PARSER="llama" ;;
    3) PARSER="hermes" ;;
    4) PARSER="mistral" ;;
    5) PARSER="" ;;
    *) echo "Invalid"; exit 1 ;;
  esac
fi

# ── Derive display name from model path ─────────────────
# mlx-community/Qwen2.5-3B-Instruct-4bit -> Qwen2.5-3B-Instruct-4bit
DISPLAY_NAME="${MODEL##*/}"

echo ""
echo "Model:  $MODEL"
echo "Parser: ${PARSER:-none}"
echo "Name:   $DISPLAY_NAME"
echo ""

# ── Write OpenCode auth ─────────────────────────────────
mkdir -p "$(dirname "$OPENCODE_AUTH")"
cat > "$OPENCODE_AUTH" << EOF
{"vllm-local": {"type": "api", "key": ""}}
EOF
echo "Wrote $OPENCODE_AUTH"

# ── Write OpenCode config ───────────────────────────────
mkdir -p "$(dirname "$OPENCODE_CONFIG")"

TOOLS_ENABLED="true"
if [ -z "$PARSER" ]; then
  TOOLS_ENABLED="false"
fi

cat > "$OPENCODE_CONFIG" << EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "vllm-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "$DISPLAY_NAME Local",
      "options": {
        "baseURL": "http://localhost:$PORT/v1"
      },
      "models": {
        "default": {
          "name": "$DISPLAY_NAME",
          "tools": $TOOLS_ENABLED,
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
EOF
echo "Wrote $OPENCODE_CONFIG"

# ── Clear OpenCode cache + stale sessions ───────────────
# Cache: provider packages (forces re-download)
if [ -d "$OPENCODE_CACHE" ]; then
  rm -rf "$OPENCODE_CACHE"
  echo "Cleared cache: $OPENCODE_CACHE"
fi

# State: sessions DB and data (keeps only auth.json we just wrote)
# Old sessions from a different model cause errors when resumed
OPENCODE_DATA="$HOME/.local/share/opencode"
if [ -d "$OPENCODE_DATA" ]; then
  find "$OPENCODE_DATA" -mindepth 1 ! -name 'auth.json' -exec rm -rf {} + 2>/dev/null || true
  echo "Cleared sessions/state in $OPENCODE_DATA (kept auth.json)"
fi

# Kill any lingering OpenCode processes
if pkill -f 'opencode' 2>/dev/null; then
  echo "Killed lingering OpenCode processes"
  sleep 1
fi

# ── Build and run server command ────────────────────────
CMD=(
  uv run vllm-mlx serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --continuous-batching
  --max-num-seqs "$MAX_NUM_SEQS"
  --prefill-batch-size "$PREFILL_BATCH_SIZE"
  --completion-batch-size "$COMPLETION_BATCH_SIZE"
  --cache-memory-percent "$CACHE_MEMORY_PERCENT"
  --stream-interval "$STREAM_INTERVAL"
  --rate-limit "$RATE_LIMIT"
  --timeout "$TIMEOUT"
  --compact-tools
  --compact-system-prompt
)

if [ -n "$PARSER" ]; then
  CMD+=(--enable-auto-tool-choice --tool-call-parser "$PARSER")
fi

echo ""
echo "Starting server..."
echo "${CMD[*]}"
echo ""
exec "${CMD[@]}"
