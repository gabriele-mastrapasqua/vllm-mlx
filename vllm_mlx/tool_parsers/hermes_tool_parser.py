# SPDX-License-Identifier: Apache-2.0
"""
Hermes/Nous tool call parser for vllm-mlx.

Handles Hermes-style tool calling format used by NousResearch models.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["hermes", "nous"])
class HermesToolParser(ToolParser):
    """
    Tool call parser for Hermes/Nous models.

    Supports multiple tool call formats:
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    - <function=name><parameter=p>v</parameter></function></tool_call>
      (no opening <tool_call> tag — common with Qwen3 models)
    - Sometimes with additional reasoning in <tool_call_reasoning>

    Used when --enable-auto-tool-choice --tool-call-parser hermes are set.
    """

    # Qwen3 / Hermes chat templates handle role="tool" and tool_calls natively.
    # Without this, tool history is converted to "[Calling tool: ...]" text,
    # which causes the model to mimic that text format instead of producing
    # proper <tool_call> XML after a few rounds of tool use.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Nemotron XML with <tool_call> wrapper:
    # <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    NEMOTRON_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    # Bare function format (no <tool_call> wrapper — common with Qwen3):
    # <function=name><parameter=p>v</parameter></function></tool_call>
    # or: <function=name><parameter=p>v</parameter></function>
    BARE_FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>(.*?)</function>\s*(?:</tool_call>)?", re.DOTALL
    )
    PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    REASONING_PATTERN = re.compile(
        r"<tool_call_reasoning>(.*?)</tool_call_reasoning>", re.DOTALL
    )

    # Markers that indicate tool call markup is being generated
    _TOOL_MARKERS = ("<tool_call>", "<function=")

    def _parse_function_matches(
        self, matches: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        """Parse function name + params_block matches into tool call dicts."""
        tool_calls = []
        for name, params_block in matches:
            params = self.PARAM_PATTERN.findall(params_block)
            arguments = {}
            for p_name, p_value in params:
                val = p_value.strip()
                try:
                    arguments[p_name.strip()] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[p_name.strip()] = val
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return tool_calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Hermes model response.
        """
        tool_calls = []
        cleaned_text = model_output

        # Remove reasoning tags first (keep for content)
        reasoning_matches = self.REASONING_PATTERN.findall(model_output)
        cleaned_text = self.REASONING_PATTERN.sub("", cleaned_text)

        # 1. Try JSON format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        matches = self.TOOL_CALL_PATTERN.findall(cleaned_text)
        for match in matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # 2. Try Nemotron XML with <tool_call> wrapper
        if not tool_calls:
            nemotron_matches = self.NEMOTRON_PATTERN.findall(cleaned_text)
            if nemotron_matches:
                tool_calls = self._parse_function_matches(nemotron_matches)
                cleaned_text = self.NEMOTRON_PATTERN.sub("", cleaned_text).strip()

        # 3. Try bare <function=name> format (no <tool_call> wrapper)
        if not tool_calls:
            bare_matches = self.BARE_FUNCTION_PATTERN.findall(cleaned_text)
            if bare_matches:
                tool_calls = self._parse_function_matches(bare_matches)
                # Clean: remove <function>...</function>, </tool_call>, and <|im_end|>
                cleaned_text = self.BARE_FUNCTION_PATTERN.sub("", cleaned_text)
                cleaned_text = cleaned_text.replace("</tool_call>", "").strip()

        # Include reasoning in content if present
        if reasoning_matches:
            reasoning_text = " ".join(reasoning_matches)
            if cleaned_text:
                cleaned_text = f"{cleaned_text}\n\n(Reasoning: {reasoning_text})"
            else:
                cleaned_text = f"(Reasoning: {reasoning_text})"

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming Hermes model output.
        """
        # Check if any tool call marker is present in the accumulated text
        has_tool_marker = any(m in current_text for m in self._TOOL_MARKERS)

        if not has_tool_marker:
            return {"content": delta_text}

        # Check if tool call is complete (closing tag present)
        # Use </function> as end marker too, since some models omit </tool_call>
        is_complete = (
            "</tool_call>" in current_text
            or ("</function>" in current_text and "<function=" in current_text)
        )

        if is_complete:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        # Still accumulating tool call markup — suppress output
        return None
