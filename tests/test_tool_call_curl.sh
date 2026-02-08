#!/usr/bin/env bash
# Test tool calling via curl against a running vllm-mlx server.
# Usage:
#   ./tests/test_tool_call_curl.sh [base_url] [api_key]
#
# Defaults:
#   base_url = http://localhost:8000
#   api_key  = (none)
#
# Start the server first, e.g.:
#   uv run vllm-mlx serve mlx-community/Qwen2.5-3B-Instruct-4bit \
#     --port 8000 --continuous-batching \
#     --enable-auto-tool-choice --tool-call-parser qwen

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-}"

AUTH_HEADER=""
if [ -n "$API_KEY" ]; then
  AUTH_HEADER="-H \"Authorization: Bearer $API_KEY\""
fi

echo "=== Test 1: Single tool (ls) ==="
echo "Expecting: tool_calls with name=ls, arguments containing path=/tmp"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d '{"model":"default","messages":[{"role":"user","content":"List files in /tmp"}],"tools":[{"type":"function","function":{"name":"ls","description":"List directory contents","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path"}},"required":["path"]}}}]}')

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Check if tool_calls is present and not null
TOOL_CALLS=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tc = data['choices'][0]['message'].get('tool_calls')
    if tc:
        print('PASS: tool_calls found')
        for call in tc:
            print(f'  -> {call[\"function\"][\"name\"]}({call[\"function\"][\"arguments\"]})')
    else:
        content = data['choices'][0]['message'].get('content', '')[:200]
        print(f'FAIL: no tool_calls. Content: {content}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "$TOOL_CALLS"
echo ""

echo "=== Test 2: Multiple tools (get_weather + get_time) ==="
echo "Expecting: tool_calls with name=get_weather"
echo ""

RESPONSE2=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d '{"model":"default","messages":[{"role":"user","content":"What is the weather in Rome?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","description":"Get current time for a timezone","parameters":{"type":"object","properties":{"timezone":{"type":"string","description":"Timezone name"}},"required":["timezone"]}}}]}')

echo "$RESPONSE2" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE2"
echo ""

TOOL_CALLS2=$(echo "$RESPONSE2" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tc = data['choices'][0]['message'].get('tool_calls')
    if tc:
        print('PASS: tool_calls found')
        for call in tc:
            print(f'  -> {call[\"function\"][\"name\"]}({call[\"function\"][\"arguments\"]})')
    else:
        content = data['choices'][0]['message'].get('content', '')[:200]
        print(f'FAIL: no tool_calls. Content: {content}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "$TOOL_CALLS2"
echo ""

echo "=== Test 3: No tools (should respond normally) ==="
echo "Expecting: normal text response, no tool_calls"
echo ""

RESPONSE3=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello in Italian"}]}')

echo "$RESPONSE3" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE3"
echo ""

TOOL_CALLS3=$(echo "$RESPONSE3" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tc = data['choices'][0]['message'].get('tool_calls')
    content = data['choices'][0]['message'].get('content', '')[:200]
    if not tc:
        print(f'PASS: no tool_calls (correct). Content: {content}')
    else:
        print(f'FAIL: unexpected tool_calls present')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "$TOOL_CALLS3"
