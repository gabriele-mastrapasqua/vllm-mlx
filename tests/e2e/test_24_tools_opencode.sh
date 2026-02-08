#!/usr/bin/env bash
# E2E: Simulate OpenCode/Claude Code sending 24 tools in one request.
# Tests that --compact-tools handles a realistic heavy payload.
#
# Usage: bash tests/e2e/test_24_tools_opencode.sh

set -uo pipefail

BASE_URL="${VLLM_MLX_URL:-http://localhost:8000}"
API_KEY="${VLLM_MLX_API_KEY:-sk-dummy-prova}"

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
bold()  { printf "\033[1m%s\033[0m\n" "$1"; }

# ── Build the full 24-tool payload (Claude Code / OpenCode style) ──
PAYLOAD=$(cat <<'ENDJSON'
{
  "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "max_tokens": 512,
  "temperature": 0,
  "messages": [
    {
      "role": "system",
      "content": "You are a coding assistant. You have access to tools. Use them when needed. When the user asks to read a file, call the Read tool."
    },
    {
      "role": "user",
      "content": "Read the file /etc/hosts and show me its contents."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "Task",
        "description": "Launch a new agent to handle complex, multi-step tasks autonomously. The Task tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.",
        "parameters": {
          "type": "object",
          "title": "TaskParams",
          "additionalProperties": false,
          "$defs": {},
          "properties": {
            "description": {"type": "string", "description": "A short (3-5 word) description of the task"},
            "prompt": {"type": "string", "description": "The task for the agent to perform"},
            "subagent_type": {"type": "string", "description": "The type of specialized agent to use for this task"},
            "model": {"type": "string", "enum": ["sonnet", "opus", "haiku"], "description": "Optional model to use for this agent"},
            "resume": {"type": "string", "description": "Optional agent ID to resume from"},
            "run_in_background": {"type": "boolean", "description": "Set to true to run this agent in the background"},
            "max_turns": {"type": "integer", "description": "Maximum number of agentic turns"}
          },
          "required": ["description", "prompt", "subagent_type"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Bash",
        "description": "Executes a given bash command with optional timeout. Working directory persists between commands; shell state (everything else) does not. The shell environment is initialized from the user's profile (bash or zsh). IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations.",
        "parameters": {
          "type": "object",
          "title": "BashParams",
          "additionalProperties": false,
          "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "description": {"type": "string", "description": "Clear, concise description of what this command does"},
            "timeout": {"type": "number", "description": "Optional timeout in milliseconds (max 600000)"},
            "run_in_background": {"type": "boolean", "description": "Set to true to run in background"}
          },
          "required": ["command"]
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
            "pattern": {"type": "string", "description": "The glob pattern to match files against"},
            "path": {"type": "string", "description": "The directory to search in. If not specified, the current working directory will be used."}
          },
          "required": ["pattern"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Grep",
        "description": "A powerful search tool built on ripgrep. Supports full regex syntax (e.g., log.*Error, function\\s+\\w+). Filter files with glob parameter (e.g., *.js, **/*.tsx) or type parameter (e.g., js, py, rust). Output modes: content shows matching lines, files_with_matches shows only file paths (default), count shows match counts.",
        "parameters": {
          "type": "object",
          "title": "GrepParams",
          "additionalProperties": false,
          "properties": {
            "pattern": {"type": "string", "description": "The regular expression pattern to search for"},
            "path": {"type": "string", "description": "File or directory to search in"},
            "glob": {"type": "string", "description": "Glob pattern to filter files"},
            "type": {"type": "string", "description": "File type to search (e.g., js, py, rust)"},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"], "description": "Output mode"},
            "-i": {"type": "boolean", "description": "Case insensitive search"},
            "-n": {"type": "boolean", "description": "Show line numbers"},
            "context": {"type": "number", "description": "Lines of context around match"},
            "-A": {"type": "number", "description": "Lines after match"},
            "-B": {"type": "number", "description": "Lines before match"},
            "head_limit": {"type": "number", "description": "Limit output to first N entries"},
            "multiline": {"type": "boolean", "description": "Enable multiline mode"}
          },
          "required": ["pattern"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Reads a file from the local filesystem. You can access any file directly by using this tool. Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.",
        "parameters": {
          "type": "object",
          "title": "ReadParams",
          "additionalProperties": false,
          "properties": {
            "file_path": {"type": "string", "description": "The absolute path to the file to read"},
            "offset": {"type": "number", "description": "The line number to start reading from"},
            "limit": {"type": "number", "description": "The number of lines to read", "default": 2000},
            "pages": {"type": "string", "description": "Page range for PDF files"}
          },
          "required": ["file_path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Edit",
        "description": "Performs exact string replacements in files. You must use your Read tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. When editing text from Read tool output, ensure you preserve the exact indentation.",
        "parameters": {
          "type": "object",
          "title": "EditParams",
          "additionalProperties": false,
          "properties": {
            "file_path": {"type": "string", "description": "The absolute path to the file to modify"},
            "old_string": {"type": "string", "description": "The text to replace"},
            "new_string": {"type": "string", "description": "The text to replace it with"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences", "default": false}
          },
          "required": ["file_path", "old_string", "new_string"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Write",
        "description": "Writes a file to the local filesystem. This tool will overwrite the existing file if there is one at the provided path. If this is an existing file, you MUST use the Read tool first to read the file's contents. ALWAYS prefer editing existing files in the codebase.",
        "parameters": {
          "type": "object",
          "title": "WriteParams",
          "additionalProperties": false,
          "properties": {
            "file_path": {"type": "string", "description": "The absolute path to the file to write"},
            "content": {"type": "string", "description": "The content to write to the file"}
          },
          "required": ["file_path", "content"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "NotebookEdit",
        "description": "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations.",
        "parameters": {
          "type": "object",
          "title": "NotebookEditParams",
          "additionalProperties": false,
          "properties": {
            "notebook_path": {"type": "string", "description": "The absolute path to the Jupyter notebook file"},
            "new_source": {"type": "string", "description": "The new source for the cell"},
            "cell_id": {"type": "string", "description": "The ID of the cell to edit"},
            "cell_type": {"type": "string", "enum": ["code", "markdown"], "description": "The type of the cell"},
            "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"], "description": "The type of edit to make"}
          },
          "required": ["notebook_path", "new_source"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "WebFetch",
        "description": "Fetches content from a specified URL and processes it using an AI model. Takes a URL and a prompt as input. Fetches the URL content, converts HTML to markdown. Processes the content with the prompt using a small, fast model.",
        "parameters": {
          "type": "object",
          "title": "WebFetchParams",
          "additionalProperties": false,
          "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from", "format": "uri"},
            "prompt": {"type": "string", "description": "The prompt to run on the fetched content"}
          },
          "required": ["url", "prompt"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "WebSearch",
        "description": "Allows Claude to search the web and use the results to inform responses. Provides up-to-date information for current events and recent data. Returns search result information formatted as search result blocks.",
        "parameters": {
          "type": "object",
          "title": "WebSearchParams",
          "additionalProperties": false,
          "properties": {
            "query": {"type": "string", "description": "The search query to use"},
            "allowed_domains": {"type": "array", "items": {"type": "string"}, "description": "Only include results from these domains"},
            "blocked_domains": {"type": "array", "items": {"type": "string"}, "description": "Never include results from these domains"}
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "ExitPlanMode",
        "description": "Use this tool when you are in plan mode and have finished writing your plan to the plan file and are ready for user approval. This tool signals that you're done planning and ready for the user to review and approve.",
        "parameters": {
          "type": "object",
          "title": "ExitPlanModeParams",
          "additionalProperties": true,
          "properties": {
            "pushToRemote": {"type": "boolean", "description": "Whether to push the plan to a remote session"},
            "remoteSessionId": {"type": "string", "description": "The remote session ID"},
            "remoteSessionUrl": {"type": "string", "description": "The remote session URL"}
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "EnterPlanMode",
        "description": "Use this tool proactively when you're about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment. This tool transitions you into plan mode where you can explore the codebase and design an implementation approach for user approval.",
        "parameters": {
          "type": "object",
          "title": "EnterPlanModeParams",
          "additionalProperties": false,
          "properties": {}
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "AskUserQuestion",
        "description": "Use this tool when you need to ask the user questions during execution. This allows you to gather user preferences or requirements, clarify ambiguous instructions, get decisions on implementation choices as you work, offer choices to the user about what direction to take.",
        "parameters": {
          "type": "object",
          "title": "AskUserQuestionParams",
          "additionalProperties": false,
          "properties": {
            "questions": {
              "type": "array",
              "description": "Questions to ask the user (1-4 questions)",
              "items": {
                "type": "object",
                "properties": {
                  "question": {"type": "string", "description": "The complete question"},
                  "header": {"type": "string", "description": "Very short label (max 12 chars)"},
                  "options": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"}
                      },
                      "required": ["label", "description"]
                    }
                  },
                  "multiSelect": {"type": "boolean", "default": false}
                },
                "required": ["question", "header", "options", "multiSelect"]
              }
            }
          },
          "required": ["questions"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "Skill",
        "description": "Execute a skill within the main conversation. When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge. When users reference a slash command or /<something>, they are referring to a skill.",
        "parameters": {
          "type": "object",
          "title": "SkillParams",
          "additionalProperties": false,
          "properties": {
            "skill": {"type": "string", "description": "The skill name"},
            "args": {"type": "string", "description": "Optional arguments for the skill"}
          },
          "required": ["skill"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskCreate",
        "description": "Use this tool to create a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. Use proactively for complex multi-step tasks, non-trivial tasks, plan mode, user-requested todo lists.",
        "parameters": {
          "type": "object",
          "title": "TaskCreateParams",
          "additionalProperties": false,
          "properties": {
            "subject": {"type": "string", "description": "A brief title for the task"},
            "description": {"type": "string", "description": "A detailed description of what needs to be done"},
            "activeForm": {"type": "string", "description": "Present continuous form shown in spinner"},
            "metadata": {"type": "object", "additionalProperties": {}, "description": "Arbitrary metadata to attach"}
          },
          "required": ["subject", "description"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskGet",
        "description": "Use this tool to retrieve a task by its ID from the task list. When you need the full description and context before starting work on a task, to understand task dependencies.",
        "parameters": {
          "type": "object",
          "title": "TaskGetParams",
          "additionalProperties": false,
          "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to retrieve"}
          },
          "required": ["taskId"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskUpdate",
        "description": "Use this tool to update a task in the task list. Mark tasks as resolved when completed. Delete tasks when no longer relevant. Update task details when requirements change.",
        "parameters": {
          "type": "object",
          "title": "TaskUpdateParams",
          "additionalProperties": false,
          "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to update"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"], "description": "New status"},
            "subject": {"type": "string", "description": "New subject"},
            "description": {"type": "string", "description": "New description"},
            "activeForm": {"type": "string", "description": "Present continuous form"},
            "owner": {"type": "string", "description": "New owner"},
            "addBlocks": {"type": "array", "items": {"type": "string"}, "description": "Task IDs this blocks"},
            "addBlockedBy": {"type": "array", "items": {"type": "string"}, "description": "Task IDs that block this"}
          },
          "required": ["taskId"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskList",
        "description": "Use this tool to list all tasks in the task list. To see what tasks are available to work on, check overall progress, find blocked tasks.",
        "parameters": {
          "type": "object",
          "title": "TaskListParams",
          "additionalProperties": false,
          "properties": {}
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskOutput",
        "description": "Retrieves output from a running or completed task (background shell, agent, or remote session). Takes a task_id parameter identifying the task. Returns the task output along with status information.",
        "parameters": {
          "type": "object",
          "title": "TaskOutputParams",
          "additionalProperties": false,
          "properties": {
            "task_id": {"type": "string", "description": "The task ID to get output from"},
            "block": {"type": "boolean", "default": true, "description": "Whether to wait for completion"},
            "timeout": {"type": "number", "default": 30000, "description": "Max wait time in ms"}
          },
          "required": ["task_id", "block", "timeout"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TaskStop",
        "description": "Stops a running background task by its ID. Takes a task_id parameter identifying the task to stop. Returns a success or failure status.",
        "parameters": {
          "type": "object",
          "title": "TaskStopParams",
          "additionalProperties": false,
          "properties": {
            "task_id": {"type": "string", "description": "The ID of the background task to stop"}
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "MultiEdit",
        "description": "Performs multiple exact string replacements in a single file atomically. All edits must match exactly once in the file unless replace_all is specified. If any edit fails to match, none of the edits are applied.",
        "parameters": {
          "type": "object",
          "title": "MultiEditParams",
          "additionalProperties": false,
          "properties": {
            "file_path": {"type": "string", "description": "The absolute path to the file to modify"},
            "edits": {
              "type": "array",
              "description": "Array of edit operations to apply",
              "items": {
                "type": "object",
                "properties": {
                  "old_string": {"type": "string", "description": "Text to find"},
                  "new_string": {"type": "string", "description": "Replacement text"},
                  "replace_all": {"type": "boolean", "default": false}
                },
                "required": ["old_string", "new_string"]
              }
            }
          },
          "required": ["file_path", "edits"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TodoRead",
        "description": "Read the current todo list from the todo file. Returns structured todo items with their IDs, descriptions, and completion status.",
        "parameters": {
          "type": "object",
          "title": "TodoReadParams",
          "additionalProperties": false,
          "properties": {}
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "TodoWrite",
        "description": "Write or update the todo list. Accepts an array of todo items with optional IDs for updating existing items.",
        "parameters": {
          "type": "object",
          "title": "TodoWriteParams",
          "additionalProperties": false,
          "properties": {
            "todos": {
              "type": "array",
              "description": "Array of todo items",
              "items": {
                "type": "object",
                "properties": {
                  "id": {"type": "string"},
                  "content": {"type": "string"},
                  "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                  "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                },
                "required": ["content", "status"]
              }
            }
          },
          "required": ["todos"]
        }
      }
    }
  ]
}
ENDJSON
)

# ── Count tools ──────────────────────────────────────────
N_TOOLS=$(echo "$PAYLOAD" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['tools']))")
bold "=== OpenCode Simulation: $N_TOOLS tools in one request ==="
echo "Server: $BASE_URL"
echo ""

# ── Measure raw JSON size ────────────────────────────────
TOOLS_SIZE=$(echo "$PAYLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(json.dumps(d['tools'])))")
echo "Raw tools JSON size: ${TOOLS_SIZE} chars"
echo ""

# ── Send request ─────────────────────────────────────────
bold "Sending request with $N_TOOLS tools..."
START=$(python3 -c "import time; print(time.time())")

RESPONSE=$(curl -s --max-time 180 -w '\n%{http_code}' \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/v1/chat/completions" \
  -d "$PAYLOAD")

END=$(python3 -c "import time; print(time.time())")
ELAPSED=$(python3 -c "print(f'{$END - $START:.2f}')")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo ""

# ── Validate response ────────────────────────────────────
if [ "$HTTP_CODE" = "200" ]; then
  green "HTTP 200 OK (${ELAPSED}s)"
else
  red "FAIL: HTTP $HTTP_CODE"
  echo "$BODY"
  exit 1
fi

# ── Parse results ────────────────────────────────────────
TMPBODY=$(mktemp)
echo "$BODY" > "$TMPBODY"
python3 - "$TMPBODY" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    body = json.load(f)

usage = body.get("usage", {})
prompt_tokens = usage.get("prompt_tokens", 0)
completion_tokens = usage.get("completion_tokens", 0)
total_tokens = usage.get("total_tokens", 0)

choice = body["choices"][0]
message = choice["message"]
finish_reason = choice.get("finish_reason", "?")

print(f"  prompt_tokens:     {prompt_tokens}")
print(f"  completion_tokens: {completion_tokens}")
print(f"  total_tokens:      {total_tokens}")
print(f"  finish_reason:     {finish_reason}")
print()

if message.get("tool_calls"):
    tc = message["tool_calls"]
    print(f"  Tool calls: {len(tc)}")
    for i, call in enumerate(tc):
        fn = call.get("function", {})
        print(f"    [{i}] {fn.get('name', '?')}({fn.get('arguments', '{}')})")
    print()
    print("\033[32m  RESULT: Model produced tool call(s) with 23 tools — compaction works!\033[0m")
elif message.get("content"):
    content = message["content"][:200]
    print(f"  Content: {content}")
    print()
    print("  RESULT: Model responded with text (no tool call). Compaction worked (no crash).")
else:
    print("  RESULT: Empty response")

PYEOF
rm -f "$TMPBODY"

echo ""
bold "Done."
