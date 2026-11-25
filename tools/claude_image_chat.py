"""Claude API bridge that routes uploaded images through ultracamnet_predict."""
import base64
import os
from typing import Any, Callable


_DEFAULT_MODEL = "claude-sonnet-4-5"
_TOOL_NAME = "ultracamnet_predict"


def caption_with_claude(
    image_bytes: bytes,
    media_type: str,
    prompt: str,
    tool_runner: Callable[[str, dict[str, str]], Any],
) -> dict[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env to use the Claude image chat.")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Claude image chat requires the optional Anthropic SDK. "
            "Install it with `python -m pip install -r requirements-claude.txt`."
        ) from exc

    model = os.environ.get("CLAUDE_IMAGE_CHAT_MODEL", _DEFAULT_MODEL).strip()
    if not model:
        raise ValueError("CLAUDE_IMAGE_CHAT_MODEL cannot be empty.")

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": encoded_image,
        },
    }
    messages = [{
        "role": "user",
        "content": [
            image_block,
            {"type": "text", "text": prompt},
        ],
    }]
    tools = [{
        "name": _TOOL_NAME,
        "description": (
            "Caption the image attached to this message using the local "
            "ultracamnet_predict image model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }]

    client = Anthropic(api_key=api_key)
    tool_response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=messages,
        tools=tools,
        tool_choice={"type": "tool", "name": _TOOL_NAME},
    )
    tool_use = next(
        (
            block for block in tool_response.content
            if block.type == "tool_use" and block.name == _TOOL_NAME
        ),
        None,
    )
    if tool_use is None:
        raise RuntimeError("Claude did not request ultracamnet_predict.")

    tool_result = tool_runner(_TOOL_NAME, {"image_base64": encoded_image})
    if not isinstance(tool_result, dict) or tool_result.get("error"):
        detail = (
            tool_result.get("error", "The image tool returned an invalid result.")
            if isinstance(tool_result, dict)
            else "The image tool returned an invalid result."
        )
        raise RuntimeError(f"ultracamnet_predict failed: {detail}")
    caption = tool_result.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise RuntimeError("ultracamnet_predict returned no caption.")

    messages.extend([
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_use.id,
                "name": tool_use.name,
                "input": tool_use.input,
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": caption,
            }],
        },
    ])
    final_response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=messages,
    )
    reply = "\n".join(
        block.text for block in final_response.content if block.type == "text"
    ).strip()
    return {"caption": caption, "reply": reply or caption}
