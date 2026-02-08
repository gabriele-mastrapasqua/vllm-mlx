#!/usr/bin/env bash
# E2E tests for --compact-tools feature.
#
# Prerequisites:
#   Server running with --compact-tools, e.g.:
#   uv run vllm-mlx serve mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
#     --port 8000 --api-key sk-dummy-prova --continuous-batching \
#     --enable-auto-tool-choice --tool-call-parser qwen \
#     --compact-tools --log-level debug
#
# Usage:
#   bash tests/e2e/test_compact_tools.sh

set -uo pipefail
# Note: not using set -e because ((PASS++)) returns exit code 1 when PASS=0

BASE_URL="${VLLM_MLX_URL:-http://localhost:8000}"
API_KEY="${VLLM_MLX_API_KEY:-sk-dummy-prova}"
PASS=0
FAIL=0

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
bold()  { printf "\033[1m%s\033[0m\n" "$1"; }

assert_ok() {
  local test_name="$1" http_code="$2"
  if [ "$http_code" = "200" ]; then
    green "  PASS: $test_name"
    ((PASS++))
  else
    red "  FAIL: $test_name (HTTP $http_code)"
    ((FAIL++))
  fi
}

assert_contains() {
  local test_name="$1" body="$2" pattern="$3"
  if echo "$body" | grep -q "$pattern"; then
    green "  PASS: $test_name"
    ((PASS++))
  else
    red "  FAIL: $test_name (pattern '$pattern' not found)"
    ((FAIL++))
  fi
}

# ── Heavy tool definitions (mimics Claude Code / OpenCode) ──────────────
# We send 5 realistic tools with verbose schemas.
# With --compact-tools, descriptions should be truncated server-side
# but tool calls should still work.
TOOLS='[
  {
    "type": "function",
    "function": {
      "name": "Read",
      "description": "Reads a file from the local filesystem. You can access any file directly by using this tool. Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid.",
      "parameters": {
        "type": "object",
        "title": "ReadParams",
        "additionalProperties": false,
        "properties": {
          "file_path": {
            "type": "string",
            "description": "The absolute path to the file to read"
          },
          "offset": {
            "type": "number",
            "description": "The line number to start reading from. Only provide if the file is too large to read at once"
          },
          "limit": {
            "type": "number",
            "description": "The number of lines to read. Only provide if the file is too large to read at once.",
            "default": 2000
          }
        },
        "required": ["file_path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "Write",
      "description": "Writes a file to the local filesystem. This tool will overwrite the existing file if there is one at the provided path. If this is an existing file, you MUST use the Read tool first to read the file contents.",
      "parameters": {
        "type": "object",
        "title": "WriteParams",
        "additionalProperties": false,
        "properties": {
          "file_path": {
            "type": "string",
            "description": "The absolute path to the file to write (must be absolute, not relative)"
          },
          "content": {
            "type": "string",
            "description": "The content to write to the file"
          }
        },
        "required": ["file_path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "Bash",
      "description": "Executes a given bash command with optional timeout. Working directory persists between commands; shell state (everything else) does not. The shell environment is initialized from the user profile.",
      "parameters": {
        "type": "object",
        "title": "BashParams",
        "additionalProperties": false,
        "properties": {
          "command": {
            "type": "string",
            "description": "The command to execute"
          },
          "timeout": {
            "type": "number",
            "description": "Optional timeout in milliseconds (max 600000)",
            "default": 120000
          }
        },
        "required": ["command"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "Grep",
      "description": "A powerful search tool built on ripgrep. Supports full regex syntax. Filter files with glob parameter or type parameter. Output modes: content shows matching lines, files_with_matches shows only file paths (default), count shows match counts.",
      "parameters": {
        "type": "object",
        "title": "GrepParams",
        "additionalProperties": false,
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The regular expression pattern to search for in file contents"
          },
          "path": {
            "type": "string",
            "description": "File or directory to search in"
          },
          "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
            "description": "Output mode"
          }
        },
        "required": ["pattern"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "Glob",
      "description": "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like **/*.js or src/**/*.ts. Returns matching file paths sorted by modification time. Use this tool when you need to find files by name patterns.",
      "parameters": {
        "type": "object",
        "title": "GlobParams",
        "additionalProperties": false,
        "properties": {
          "pattern": {
            "type": "string",
            "description": "The glob pattern to match files against"
          },
          "path": {
            "type": "string",
            "description": "The directory to search in"
          }
        },
        "required": ["pattern"]
      }
    }
  }
]'

# ═══════════════════════════════════════════════════════════
bold "=== E2E: Compact Tools Tests ==="
echo "Server: $BASE_URL"
echo ""

# ── Test 1: Health check ──────────────────────────────────
bold "Test 1: Server health"
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/health")
assert_ok "GET /health returns 200" "$HTTP_CODE"

# ── Test 2: Chat with tools (non-streaming) ───────────────
bold "Test 2: Chat with 5 tools — non-streaming tool call"
RESPONSE=$(curl -s --max-time 120 -w '\n%{http_code}' \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/v1/chat/completions" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "max_tokens": 256,
    "temperature": 0,
    "messages": [
      {"role": "system", "content": "You are a coding assistant. Use tools when needed."},
      {"role": "user", "content": "Read the file /etc/hostname"}
    ],
    "tools": '"$TOOLS"'
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')
assert_ok "POST /v1/chat/completions returns 200" "$HTTP_CODE"
assert_contains "Response has choices" "$BODY" '"choices"'
# Check if model produced a tool call (ideal) or at least a response
if echo "$BODY" | grep -q '"tool_calls"'; then
  green "  PASS: Model produced tool_calls"
  ((PASS++))
  assert_contains "Tool call targets Read" "$BODY" '"Read"'
else
  echo "  INFO: Model responded with text instead of tool call (may happen with small models)"
  assert_contains "Response has content" "$BODY" '"content"'
fi

# Check prompt_tokens is reasonable (compacted should be lower)
PROMPT_TOKENS=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "0")
echo "  INFO: prompt_tokens=$PROMPT_TOKENS (with compaction)"

# ── Test 3: Chat with tools (streaming) ───────────────────
bold "Test 3: Chat with 5 tools — streaming"
STREAM_RESPONSE=$(curl -s --max-time 120 \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/v1/chat/completions" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "max_tokens": 256,
    "temperature": 0,
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a coding assistant. Use tools."},
      {"role": "user", "content": "List files in /tmp using Bash"}
    ],
    "tools": '"$TOOLS"'
  }')

assert_contains "Streaming returns data chunks" "$STREAM_RESPONSE" "data:"
assert_contains "Streaming ends with [DONE]" "$STREAM_RESPONSE" "[DONE]"

# ── Test 4: Anthropic /v1/messages with tools ─────────────
bold "Test 4: Anthropic /v1/messages with tools"
ANTHROPIC_TOOLS='[
  {
    "name": "Read",
    "description": "Reads a file from the local filesystem. You can access any file directly by using this tool. Assume this tool is able to read all files on the machine.",
    "input_schema": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string", "description": "The absolute path to the file to read"}
      },
      "required": ["file_path"]
    }
  },
  {
    "name": "Bash",
    "description": "Executes a given bash command with optional timeout. Working directory persists between commands.",
    "input_schema": {
      "type": "object",
      "properties": {
        "command": {"type": "string", "description": "The command to execute"}
      },
      "required": ["command"]
    }
  }
]'

ANTH_RESPONSE=$(curl -s --max-time 120 -w '\n%{http_code}' \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  "$BASE_URL/v1/messages" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "max_tokens": 256,
    "temperature": 0,
    "messages": [
      {"role": "user", "content": "Read the file /etc/hosts"}
    ],
    "tools": '"$ANTHROPIC_TOOLS"'
  }')

ANTH_CODE=$(echo "$ANTH_RESPONSE" | tail -1)
ANTH_BODY=$(echo "$ANTH_RESPONSE" | sed '$d')
assert_ok "POST /v1/messages returns 200" "$ANTH_CODE"
assert_contains "Anthropic response has content" "$ANTH_BODY" '"content"'

# ── Test 5: Chat WITHOUT tools (baseline — no compaction) ─
bold "Test 5: Chat without tools (baseline, no compaction applied)"
NOTOOL_RESPONSE=$(curl -s --max-time 120 -w '\n%{http_code}' \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/v1/chat/completions" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "max_tokens": 64,
    "temperature": 0,
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ]
  }')

NOTOOL_CODE=$(echo "$NOTOOL_RESPONSE" | tail -1)
NOTOOL_BODY=$(echo "$NOTOOL_RESPONSE" | sed '$d')
assert_ok "Chat without tools returns 200" "$NOTOOL_CODE"
assert_contains "Has content in response" "$NOTOOL_BODY" '"content"'

# ── Summary ───────────────────────────────────────────────
echo ""
bold "=== Results ==="
green "Passed: $PASS"
if [ "$FAIL" -gt 0 ]; then
  red "Failed: $FAIL"
  exit 1
else
  echo "Failed: 0"
  green "All tests passed!"
fi
