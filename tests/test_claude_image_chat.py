import base64
import sys
import types
from types import SimpleNamespace

import pytest

from tools.claude_image_chat import caption_with_claude


def test_claude_image_chat_forces_ultracam_tool_and_returns_reply(monkeypatch):
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(
                    type="tool_use",
                    id="tool-call-1",
                    name="ultracamnet_predict",
                    input={},
                )])
            return SimpleNamespace(content=[SimpleNamespace(
                type="text",
                text="The image shows a small dog.",
            )])

    class FakeAnthropic:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.messages = FakeMessages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tool_calls = []

    result = caption_with_claude(
        b"jpeg bytes",
        "image/jpeg",
        "Caption this image.",
        lambda name, arguments: tool_calls.append((name, arguments)) or {
            "caption": "a small dog"
        },
    )

    encoded_image = base64.b64encode(b"jpeg bytes").decode("ascii")
    assert result == {
        "caption": "a small dog",
        "reply": "The image shows a small dog.",
    }
    assert calls[0]["tool_choice"] == {
        "type": "tool", "name": "ultracamnet_predict",
    }
    assert calls[0]["messages"][0]["content"][0]["source"]["data"] == encoded_image
    assert tool_calls == [(
        "ultracamnet_predict", {"image_base64": encoded_image},
    )]
    assert calls[1]["messages"][-1]["content"][0]["content"] == "a small dog"


def test_claude_image_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set ANTHROPIC_API_KEY"):
        caption_with_claude(b"image", "image/jpeg", "Caption it.", lambda *_: {})
