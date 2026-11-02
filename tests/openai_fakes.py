"""Lightweight fake OpenAI chat.completions responses for offline tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock


@dataclass
class FakeFunction:
    name: str
    arguments: str = "{}"


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class FakeCompletion:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)


def completion_with_content(content: str) -> FakeCompletion:
    return FakeCompletion(choices=[FakeChoice(message=FakeMessage(content=content))])


def completion_with_json(payload: dict[str, Any]) -> FakeCompletion:
    return completion_with_content(json.dumps(payload))


def completion_with_tool_calls(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    *,
    content: str | None = None,
) -> FakeCompletion:
    """tool_calls: list of (call_id, function_name, arguments_dict)."""
    calls = [
        FakeToolCall(
            id=call_id,
            function=FakeFunction(name=name, arguments=json.dumps(args)),
        )
        for call_id, name, args in tool_calls
    ]
    return FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(content=content, tool_calls=calls))]
    )


def make_async_openai_client(create_side_effect: Any) -> MagicMock:
    """Build a MagicMock AsyncOpenAI whose chat.completions.create is AsyncMock."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    return client
