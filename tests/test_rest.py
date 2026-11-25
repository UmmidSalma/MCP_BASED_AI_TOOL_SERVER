import base64
from io import BytesIO

import pytest

from server import _handlers
from security.validation import TOOL_INPUTS
from ui_server import _MAX_IMAGE_BYTES, app


def test_rest_tools_and_validation():
    client = app.test_client()
    tool_names = client.get("/api/tools").json["tools"]
    assert set(tool_names) == set(TOOL_INPUTS) == set(_handlers)
    assert client.post("/api/tools/ping", json={}).json["result"] == "pong"
    response = client.post("/api/tools/fraud_risk_analyzer", json={
        "amount": -1, "hour": 12, "country": "IN",
    })
    assert response.status_code == 400


def test_unknown_rest_tool_is_not_callable():
    response = app.test_client().post("/api/tools/not_a_tool", json={})
    assert response.status_code == 404


def test_rest_dashboard_displays_tool_activity_and_metrics():
    client = app.test_client()
    assert client.post("/api/tools/ping", json={}).status_code == 200

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert b'id="image-form"' in dashboard.data
    assert b"Recent calls" in dashboard.data
    assert b"Avg ms" in dashboard.data
    assert b"max_latency_ms" in dashboard.data

    logs_response = client.get("/api/logs")
    assert logs_response.status_code == 200
    assert logs_response.json["calls"][0]["tool"] == "ping"
    metrics_response = client.get("/api/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json["tools"]["ping"]["successes"] >= 1


def test_rest_requires_json_object_body():
    response = app.test_client().post(
        "/api/tools/ping",
        data="not json",
        content_type="application/json",
    )
    assert response.status_code == 400


def test_rest_ultracamnet_returns_invalid_base64_as_tool_error():
    response = app.test_client().post(
        "/api/tools/ultracamnet_predict",
        json={"image_base64": "not-base64"},
    )

    assert response.status_code == 200
    assert "valid standard base64" in response.json["result"]["error"]


def test_dashboard_image_upload_uses_uploaded_bytes(monkeypatch):
    received = {}

    def fake_call_tool(_tool_name, arguments):
        received.update(arguments)
        return {"caption": "uploaded image"}

    monkeypatch.setattr("ui_server.call_tool", fake_call_tool)
    response = app.test_client().post(
        "/api/tools/ultracamnet_predict",
        data={"image": (BytesIO(b"image bytes"), "screenshot.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["result"] == {"caption": "uploaded image"}
    assert base64.b64decode(received["image_base64"]) == b"image bytes"


def test_dashboard_rejects_oversized_image_upload():
    response = app.test_client().post(
        "/api/tools/ultracamnet_predict",
        data={"image": (BytesIO(b"x" * (_MAX_IMAGE_BYTES + 1)), "large.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.json == {"error": "Image exceeds the 5 MB size limit."}


def test_claude_chat_upload_calls_bridge(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    image_buffer = BytesIO()
    Image.new("RGB", (4, 4), (30, 60, 90)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    bridge_args = {}

    def fake_bridge(image, media_type, prompt, tool_runner):
        bridge_args.update(
            image=image,
            media_type=media_type,
            prompt=prompt,
            tool_runner=tool_runner,
        )
        return {"caption": "a test image", "reply": "This is a test image."}

    monkeypatch.setattr("ui_server.caption_with_claude", fake_bridge)
    response = app.test_client().post(
        "/api/claude-image-chat",
        data={
            "image": (BytesIO(image_bytes), "test.png"),
            "message": "Please caption this.",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["result"]["caption"] == "a test image"
    assert bridge_args["image"] == image_bytes
    assert bridge_args["media_type"] == "image/png"
    assert bridge_args["prompt"] == "Please caption this."
    assert bridge_args["tool_runner"] is not None