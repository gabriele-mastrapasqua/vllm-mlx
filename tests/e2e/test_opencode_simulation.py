"""
E2E test simulating OpenCode client behavior.

Sends requests with a realistic 10K system prompt + 24 tools,
verifying that --compact-tools and --compact-system-prompt
allow a 7B model to produce correct tool calls.

Prerequisites:
    Server running with compaction flags:
    uv run vllm-mlx serve mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
      --port 8000 --api-key sk-dummy-prova --continuous-batching \
      --enable-auto-tool-choice --tool-call-parser qwen \
      --compact-tools --compact-tools-level aggressive \
      --compact-system-prompt --log-level debug

Usage:
    pytest tests/e2e/test_opencode_simulation.py -v
    # or directly:
    python tests/e2e/test_opencode_simulation.py
"""

import json
import os
import urllib.request
import urllib.error

BASE_URL = os.environ.get("VLLM_MLX_URL", "http://localhost:8000")
API_KEY = os.environ.get("VLLM_MLX_API_KEY", "sk-dummy-prova")

# ── Realistic OpenCode system prompt (~10K chars) ────────────────────────

OPENCODE_SYSTEM_PROMPT = (
    "You are opencode, an interactive CLI tool that helps users with "
    "software engineering tasks. Use the instructions below and the tools "
    "available to you to assist the user.\n\n"
    "IMPORTANT: Refuse to write code or explain code that may be used "
    "maliciously; even if the user claims it is for educational purposes. "
    "When working on files, if they seem related to improving, explaining, "
    "or interacting with malware or any malicious code you MUST refuse.\n"
    "IMPORTANT: Before you begin work, think about what the code does at a "
    "high level. If it seems malicious, refuse.\n\n"
    "# Doing tasks\n"
    "The user will primarily request you to perform software engineering "
    "tasks. These may include solving bugs, adding new functionality, "
    "refactoring code, explaining code, and more.\n\n"
    "# Using tools\n"
    "You have access to tools to help complete tasks. Use the appropriate "
    "tool for each action. Do not simulate tool output.\n\n"
    # Pad to ~10K chars like real OpenCode
    + "# Additional Guidelines\n"
    + (
        "When working on tasks, follow best practices for software engineering. "
        "Write clean, maintainable code. Use meaningful variable names. "
        "Add comments where the logic is not obvious. Handle errors gracefully. "
        "Consider edge cases and boundary conditions. Write tests when appropriate. "
        "Follow the existing code style and conventions of the project. "
        "Do not introduce breaking changes unless explicitly asked. "
        "Keep changes minimal and focused on the task at hand. "
    )
    * 20
)

# ── 24 tools matching OpenCode/Claude Code format ───────────────────────

OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params,
        },
    }
    for name, desc, params in [
        (
            "Bash",
            "Executes a given bash command with optional timeout. Working directory "
            "persists between commands; shell state does not. The shell environment is "
            "initialized from the user's profile (bash or zsh). IMPORTANT: This tool is "
            "for terminal operations like git, npm, docker, etc.",
            {
                "type": "object",
                "title": "BashParams",
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear, concise description of what this command does",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in milliseconds (max 600000)",
                        "default": 120000,
                    },
                },
                "required": ["command"],
            },
        ),
        (
            "Read",
            "Reads a file from the local filesystem. You can access any file directly "
            "by using this tool. Assume this tool is able to read all files on the machine. "
            "If the User provides a path to a file assume that path is valid.",
            {
                "type": "object",
                "title": "ReadParams",
                "additionalProperties": False,
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read",
                    },
                    "offset": {
                        "type": "number",
                        "description": "The line number to start reading from",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of lines to read",
                        "default": 2000,
                    },
                },
                "required": ["file_path"],
            },
        ),
        (
            "Write",
            "Writes a file to the local filesystem. This tool will overwrite the existing "
            "file if there is one at the provided path. If this is an existing file, you "
            "MUST use the Read tool first to read the file's contents.",
            {
                "type": "object",
                "title": "WriteParams",
                "additionalProperties": False,
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
        ),
        (
            "Edit",
            "Performs exact string replacements in files. You must use your Read tool at "
            "least once before editing. Preserves exact indentation.",
            {
                "type": "object",
                "title": "EditParams",
                "additionalProperties": False,
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file",
                    },
                    "old_string": {"type": "string", "description": "Text to replace"},
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Replace all occurrences",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        (
            "Glob",
            "Fast file pattern matching tool that works with any codebase size. Supports "
            "glob patterns like **/*.js or src/**/*.ts. Returns matching file paths sorted "
            "by modification time.",
            {
                "type": "object",
                "title": "GlobParams",
                "additionalProperties": False,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern"],
            },
        ),
        (
            "Grep",
            "A powerful search tool built on ripgrep. Supports full regex syntax. Filter "
            "files with glob or type parameter. Output modes: content, files_with_matches, count.",
            {
                "type": "object",
                "title": "GrepParams",
                "additionalProperties": False,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode",
                    },
                },
                "required": ["pattern"],
            },
        ),
        (
            "WebFetch",
            "Fetches content from a URL and processes it. Takes a URL and a prompt. "
            "Converts HTML to markdown and processes with a small model.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to run on content",
                    },
                },
                "required": ["url", "prompt"],
            },
        ),
        (
            "WebSearch",
            "Search the web for up-to-date information. Returns search results with links.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        ),
        (
            "Task",
            "Launch a new agent for complex multi-step tasks. Each agent type has "
            "specific capabilities.",
            {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Short description"},
                    "prompt": {"type": "string", "description": "Task prompt"},
                    "subagent_type": {"type": "string", "description": "Agent type"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        ),
        (
            "TaskCreate",
            "Create a structured task list for tracking progress.",
            {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Details"},
                },
                "required": ["subject", "description"],
            },
        ),
        (
            "TaskGet",
            "Retrieve a task by ID from the task list.",
            {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string", "description": "Task ID"},
                },
                "required": ["taskId"],
            },
        ),
        (
            "TaskUpdate",
            "Update a task in the task list. Mark resolved, change details, set dependencies.",
            {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string", "description": "Task ID"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "deleted"],
                    },
                },
                "required": ["taskId"],
            },
        ),
        (
            "TaskList",
            "List all tasks in the task list.",
            {"type": "object", "properties": {}},
        ),
        (
            "TaskOutput",
            "Retrieve output from a running or completed background task.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "block": {"type": "boolean", "default": True},
                    "timeout": {"type": "number", "default": 30000},
                },
                "required": ["task_id", "block", "timeout"],
            },
        ),
        (
            "TaskStop",
            "Stop a running background task by ID.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to stop"},
                },
            },
        ),
        (
            "Skill",
            "Execute a skill (slash command) within the conversation.",
            {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "Skill name"},
                    "args": {"type": "string", "description": "Arguments"},
                },
                "required": ["skill"],
            },
        ),
        (
            "AskUserQuestion",
            "Ask the user questions during execution to gather preferences or clarify.",
            {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Questions to ask (1-4)",
                    },
                },
                "required": ["questions"],
            },
        ),
        (
            "EnterPlanMode",
            "Enter plan mode for non-trivial implementation tasks.",
            {"type": "object", "properties": {}},
        ),
        (
            "ExitPlanMode",
            "Exit plan mode after writing plan, ready for user approval.",
            {"type": "object", "additionalProperties": True, "properties": {}},
        ),
        (
            "NotebookEdit",
            "Replace contents of a Jupyter notebook cell.",
            {
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Path to notebook",
                    },
                    "new_source": {
                        "type": "string",
                        "description": "New cell source",
                    },
                },
                "required": ["notebook_path", "new_source"],
            },
        ),
        (
            "MultiEdit",
            "Multiple exact string replacements in a single file atomically.",
            {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File path"},
                    "edits": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Edit operations",
                    },
                },
                "required": ["file_path", "edits"],
            },
        ),
        (
            "TodoRead",
            "Read the current todo list.",
            {"type": "object", "properties": {}},
        ),
        (
            "TodoWrite",
            "Write or update the todo list.",
            {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Todo items",
                    },
                },
                "required": ["todos"],
            },
        ),
        (
            "ToolSearch",
            "Search for available MCP tools by name or description.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        ),
    ]
]


def _post(payload: dict, timeout: int = 120) -> dict:
    """Send a request to the server and return parsed JSON."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


def test_health():
    """Server is healthy."""
    req = urllib.request.Request(f"{BASE_URL}/health")
    resp = urllib.request.urlopen(req, timeout=5)
    body = json.loads(resp.read())
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_greeting_no_tools():
    """Simple greeting without tools — model responds with text."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 128,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "ciao"},
            ],
        }
    )
    msg = body["choices"][0]["message"]
    assert msg.get("content"), "Expected text content for greeting"
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    print(f"  greeting response: {msg['content'][:100]!r}")


def test_opencode_greeting_with_tools():
    """OpenCode greeting with 24 tools — model should respond with text (no tool needed)."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 128,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "ciao"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg = body["choices"][0]["message"]
    prompt_tokens = body["usage"]["prompt_tokens"]
    print(f"  prompt_tokens: {prompt_tokens}")
    # With compaction, should be well under 3000 tokens
    assert prompt_tokens < 3000, f"prompt_tokens too high: {prompt_tokens}"
    # Model may respond with text or (incorrectly) a tool call for a greeting
    # Either is acceptable — the key is that the server didn't crash
    assert msg.get("content") or msg.get("tool_calls"), "Expected some response"
    if msg.get("content"):
        print(f"  response: {msg['content'][:100]!r}")


def test_opencode_ls_tool_call():
    """OpenCode 'ls -la .' with 24 tools — model MUST produce Bash tool call."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "fammi ls -la ."},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg = body["choices"][0]["message"]
    prompt_tokens = body["usage"]["prompt_tokens"]
    finish_reason = body["choices"][0]["finish_reason"]

    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  finish_reason: {finish_reason}")

    assert prompt_tokens < 3000, f"prompt_tokens too high: {prompt_tokens}"

    # Model should produce a tool call
    tool_calls = msg.get("tool_calls")
    assert tool_calls, (
        f"Expected tool_calls but got text: {(msg.get('content') or '')[:200]!r}"
    )
    assert finish_reason == "tool_calls"

    fn = tool_calls[0]["function"]
    print(f"  tool_call: {fn['name']}({fn['arguments']})")
    assert fn["name"] == "Bash", f"Expected Bash tool, got {fn['name']}"
    args = json.loads(fn["arguments"])
    assert "ls" in args.get("command", args.get("input", "")), (
        f"Expected 'ls' in arguments: {fn['arguments']}"
    )


def test_opencode_read_file_tool_call():
    """OpenCode 'leggi /etc/hosts' — model MUST produce Read tool call."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "leggi il file /etc/hosts"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg = body["choices"][0]["message"]
    prompt_tokens = body["usage"]["prompt_tokens"]

    print(f"  prompt_tokens: {prompt_tokens}")
    assert prompt_tokens < 3000

    tool_calls = msg.get("tool_calls")
    assert tool_calls, (
        f"Expected tool_calls but got text: {(msg.get('content') or '')[:200]!r}"
    )

    fn = tool_calls[0]["function"]
    print(f"  tool_call: {fn['name']}({fn['arguments']})")
    assert fn["name"] == "Read", f"Expected Read tool, got {fn['name']}"


def test_opencode_multi_turn_with_tools():
    """Multi-turn: greeting then tool call — simulates real OpenCode conversation."""
    # Turn 1: greeting (assistant responds with text)
    body1 = _post(
        {
            "model": "default",
            "max_tokens": 128,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "ciao"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    assistant_reply = body1["choices"][0]["message"].get("content", "Ciao!")
    print(f"  turn 1 reply: {assistant_reply[:80]!r}")

    # Turn 2: user asks for ls (should produce tool call)
    body2 = _post(
        {
            "model": "default",
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "ciao"},
                {"role": "assistant", "content": assistant_reply},
                {"role": "user", "content": "fammi ls -la ."},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg2 = body2["choices"][0]["message"]
    prompt_tokens = body2["usage"]["prompt_tokens"]
    print(f"  turn 2 prompt_tokens: {prompt_tokens}")

    tool_calls = msg2.get("tool_calls")
    content = msg2.get("content") or ""
    if tool_calls:
        fn = tool_calls[0]["function"]
        print(f"  turn 2 tool_call: {fn['name']}({fn['arguments']})")
        assert fn["name"] == "Bash"
    else:
        # Small models sometimes hallucinate output in multi-turn;
        # acceptable as long as the model gave *some* response
        print(f"  turn 2 text (no tool call): {content[:150]!r}")
        assert len(content) > 5, f"Expected some response on turn 2, got: {content!r}"


def test_general_knowledge_no_fake_tools():
    """General knowledge question — model should answer in text, NOT invent tools."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "cos'è la teoria della relatività?"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    print(f"  content: {content[:150]!r}")
    if tool_calls:
        fn = tool_calls[0]["function"]
        print(f"  tool_call: {fn['name']}({fn.get('arguments', '')})")
        # If the model calls a tool, it must be a real one (not invented)
        real_tool_names = {t["function"]["name"] for t in OPENCODE_TOOLS}
        assert fn["name"] in real_tool_names, (
            f"Model invented a fake tool: {fn['name']!r}. "
            f"Should answer in text or use a real tool."
        )
    else:
        # Best case: model answered directly in text
        assert len(content) > 20, f"Expected a real answer, got: {content!r}"
        print("  OK: model answered in text without calling tools")


def test_web_search_not_invented():
    """When asked to search the web, model should NOT call fake/hallucinated tools."""
    body = _post(
        {
            "model": "default",
            "max_tokens": 256,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "cerca sul web le ultime news su python 3.13"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    print(f"  content: {content[:150]!r}")
    if tool_calls:
        fn = tool_calls[0]["function"]
        print(f"  tool_call: {fn['name']}({fn.get('arguments', '')})")
        real_tool_names = {t["function"]["name"] for t in OPENCODE_TOOLS}
        assert fn["name"] in real_tool_names, (
            f"Model invented a fake tool: {fn['name']!r}. "
            f"Valid tools: {sorted(real_tool_names)}"
        )
        # WebSearch is a real tool in the list, so calling it is OK
        if fn["name"] == "WebSearch":
            print("  OK: model correctly used WebSearch tool")
    else:
        # Answering in text is also acceptable
        assert len(content) > 10, f"Expected some response, got: {content!r}"
        print("  OK: model answered in text")


def test_token_savings():
    """Verify compaction provides significant token savings."""
    # Minimal request to measure baseline vs compacted
    body = _post(
        {
            "model": "default",
            "max_tokens": 64,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": OPENCODE_SYSTEM_PROMPT},
                {"role": "user", "content": "hi"},
            ],
            "tools": OPENCODE_TOOLS,
        }
    )
    prompt_tokens = body["usage"]["prompt_tokens"]
    print(f"  prompt_tokens with 24 tools + 10K system: {prompt_tokens}")
    # Without compaction this would be 4000-6000+ tokens
    # With both compactions it should be under 2000
    assert prompt_tokens < 2000, (
        f"Token savings insufficient: {prompt_tokens} tokens "
        "(expected <2000 with full compaction)"
    )


# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tests = [
        test_health,
        test_greeting_no_tools,
        test_opencode_greeting_with_tools,
        test_opencode_ls_tool_call,
        test_opencode_read_file_tool_call,
        test_opencode_multi_turn_with_tools,
        test_general_knowledge_no_fake_tools,
        test_web_search_not_invented,
        test_token_savings,
    ]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            print(f"\n{name}...")
            test()
            print(f"  \033[32mPASS\033[0m")
            passed += 1
        except Exception as e:
            print(f"  \033[31mFAIL: {e}\033[0m")
            failed += 1

    print(f"\n{'='*50}")
    print(f"\033[32mPassed: {passed}\033[0m")
    if failed:
        print(f"\033[31mFailed: {failed}\033[0m")
        sys.exit(1)
    else:
        print("Failed: 0")
        print("\033[32mAll tests passed!\033[0m")
