import json
import ssl
import sys

import pytest
from pydantic import ValidationError

from security.validation import validate_image_file
from server import call_tool
from tools import external, fraud, integrations


def test_fraud_tool_requires_model(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(fraud, "_model", None)
    monkeypatch.setattr(fraud, "_loaded_path", None)
    with pytest.raises(FileNotFoundError, match="Fraud model not found"):
        fraud.score_transaction(100, 12, "IN")


def test_fraud_tool_reports_missing_optional_dependency(monkeypatch, tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text("{}")
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(model_path))
    monkeypatch.setattr(fraud, "_model", None)
    monkeypatch.setattr(fraud, "_explainer", None)
    monkeypatch.setattr(fraud, "_loaded_path", None)
    monkeypatch.setitem(sys.modules, "shap", None)

    with pytest.raises(RuntimeError, match="missing 'shap'") as error:
        fraud.score_transaction(100, 12, "IN")

    assert "python -m pip install -r requirements-ml.txt" in str(error.value)


def test_relative_fraud_model_path_uses_project_root(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRAUD_MODEL_PATH", raising=False)
    monkeypatch.setattr(fraud, "_PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(fraud, "_model", None)
    monkeypatch.setattr(fraud, "_loaded_path", None)

    with pytest.raises(FileNotFoundError) as error:
        fraud._load_model()

    expected = tmp_path / "project" / "models" / "fraud_xgb.json"
    assert str(expected) in str(error.value)


def test_fraud_tool_returns_model_probability_and_shap(monkeypatch):
    class Model:
        classes_ = [0, 1]

        def get_booster(self):
            return type("Booster", (), {"feature_names": None})()

        def predict_proba(self, row):
            assert len(row[0]) == 4
            return [[0.15, 0.85]]

    class Explainer:
        def shap_values(self, row):
            return [[0.1, -0.5, 0.2, 0.3]]

    monkeypatch.setattr(fraud, "_load_model", lambda: (Model(), Explainer()))
    result = fraud.score_transaction(120, 14, "IN", True)
    assert result["risk_score"] == 0.85
    assert result["label"] == "HIGH"
    assert result["reasons"][0]["feature"] == "hour"
    assert result["reasons"][0]["direction"] == "decreases risk"


def test_dispatch_rejects_out_of_range_input():
    with pytest.raises(ValidationError):
        call_tool("fraud_risk_analyzer", {"amount": 10, "hour": 24, "country": "IN"})


def test_image_path_must_stay_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "escape.png"
    outside.write_bytes(b"not actually decoded here")
    monkeypatch.setenv("IMAGE_ROOT", str(root))
    with pytest.raises(ValueError, match="inside"):
        validate_image_file(str(outside))


def test_image_path_rejects_symlink_escape(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "escape.png"
    outside.write_bytes(b"image")
    link = root / "linked.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not available.")

    monkeypatch.setenv("IMAGE_ROOT", str(root))
    with pytest.raises(ValueError, match="inside"):
        validate_image_file(str(link))


def test_image_path_enforces_size_limit(tmp_path, monkeypatch):
    image = tmp_path / "large.png"
    image.write_bytes(b"12345")
    monkeypatch.setenv("IMAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("MAX_IMAGE_BYTES", "4")
    with pytest.raises(ValueError, match="size limit"):
        validate_image_file(str(image), max_bytes=4)


def test_wikipedia_summary_uses_mocked_api(monkeypatch):
    monkeypatch.setattr(external, "_get_json", lambda url: {
        "title": "Example", "extract": "A short summary.",
        "content_urls": {"desktop": {"page": "https://example.test"}},
    })
    assert external.wikipedia_summary("Example") == {
        "title": "Example", "summary": "A short summary.", "url": "https://example.test",
    }


def test_arxiv_search_parses_mocked_feed(monkeypatch):
    feed = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <title>Paper title</title><id>https://arxiv.org/abs/1234</id>
      <summary>Paper summary</summary></entry></feed>'''
    monkeypatch.setattr(external, "_get", lambda _url: feed)
    assert external.arxiv_search("paper") == [{
        "title": "Paper title", "link": "https://arxiv.org/abs/1234", "summary": "Paper summary",
    }]


def test_external_http_uses_verified_ca_context(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"response"

    def fake_urlopen(_request, timeout, context):
        captured["timeout"] = timeout
        captured["context"] = context
        return Response()

    monkeypatch.setattr(external.urllib.request, "urlopen", fake_urlopen)

    assert external._get("https://example.test") == b"response"
    assert captured["timeout"] == 15
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED
    assert captured["context"].check_hostname is True
    assert captured["context"].get_ca_certs()


def test_github_summary_uses_mocked_api(monkeypatch):
    monkeypatch.setattr(external, "_get_json", lambda _url: {
        "full_name": "owner/repo", "description": "A project", "stargazers_count": 4,
        "forks_count": 2, "language": "Python", "open_issues_count": 1,
        "default_branch": "main", "html_url": "https://github.com/owner/repo",
    })
    assert external.github_repo_summary("owner/repo")["stars"] == 4


def test_github_account_profile_and_repositories(monkeypatch):
    responses = iter([
        {"login": "UmmidSalma", "name": "Salma", "public_repos": 2, "followers": 3,
         "following": 1, "html_url": "https://github.com/UmmidSalma"},
        [{"name": "project", "full_name": "UmmidSalma/project", "stargazers_count": 5,
          "forks_count": 1, "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/UmmidSalma/project"}],
    ])
    monkeypatch.setattr(external, "_get_json", lambda _url: next(responses))
    profile = external.github_account_summary("@UmmidSalma")
    repos = external.github_list_repositories("UmmidSalma", 10)
    assert profile["public_repositories"] == 2
    assert repos["repositories"][0]["full_name"] == "UmmidSalma/project"


def test_github_repo_context_returns_bounded_readme(monkeypatch):
    import base64

    responses = iter([
        {"full_name": "owner/repo", "description": "Project", "topics": ["demo"],
         "stargazers_count": 1, "html_url": "https://github.com/owner/repo"},
        {"content": base64.b64encode(b"# Project\nUseful README").decode()},
    ])
    monkeypatch.setattr(external, "_get_json", lambda _url: next(responses))
    context = external.github_repo_context("owner/repo")
    assert context["repository"]["name"] == "owner/repo"
    assert context["readme"].startswith("# Project")
    assert not context["readme_truncated"]


def test_weather_uses_mocked_geocoding_and_forecast(monkeypatch):
    responses = iter([
        {"results": [{"name": "Toronto", "country": "Canada", "latitude": 43.7, "longitude": -79.4}]},
        {"timezone": "America/Toronto", "current": {"temperature_2m": 18}},
    ])
    monkeypatch.setattr(external, "_get_json", lambda _url: next(responses))
    result = external.weather_forecast("Toronto")
    assert result["temperature_c"] == 18
    assert result["country"] == "Canada"


def test_currency_conversion_uses_mocked_rates(monkeypatch):
    monkeypatch.setattr(external, "_get_json", lambda _url: {
        "rates": {"EUR": 0.9}, "provider": "example",
    })
    result = external.currency_converter(10, "usd", "eur")
    assert result["converted_amount"] == 9
    assert result["from_currency"] == "USD"


def test_news_headlines_uses_mocked_api(monkeypatch):
    monkeypatch.setattr(external, "_get_json", lambda _url: {
        "hits": [{"title": "Story", "url": "https://example.test", "author": "reader", "points": 2,
                  "created_at": "2026-01-01T00:00:00Z"}],
    })
    assert external.news_headlines("topic", 1)[0]["title"] == "Story"


def test_google_calendar_tool_uses_mocked_http(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"items":[{"summary":"Meeting"}]}'

    monkeypatch.setattr(integrations, "access_token", lambda _provider: "test-token")
    monkeypatch.setattr(integrations.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert integrations.google_calendar_events()["events"][0]["summary"] == "Meeting"


def test_notion_tool_uses_mocked_http(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"results":[{"id":"page-1"}]}'

    monkeypatch.setattr(integrations, "access_token", lambda _provider: "test-token")
    monkeypatch.setattr(integrations.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert integrations.notion_search("notes")["results"][0]["id"] == "page-1"


def test_rate_limit_and_audit_event(monkeypatch):
    from security import observability

    observability._call_times["ping"].clear()
    monkeypatch.setenv("RATE_LIMIT_CALLS", "1")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    assert call_tool("ping", {}) == "pong"
    with pytest.raises(RuntimeError, match="Rate limit"):
        call_tool("ping", {})
    success_event, failure_event = observability.recent_calls(2)[::-1]
    assert success_event["tool"] == "ping"
    assert len(success_event["args_hash"]) == 64
    assert "arguments" not in success_event
    assert success_event["status"] == "success"
    assert success_event["success"] is True
    assert failure_event["status"] == "failure"
    assert failure_event["success"] is False


def test_failed_tool_call_is_audited_and_counted(monkeypatch, caplog):
    from security import observability

    observability._call_times["wikipedia_summary"].clear()
    observability._events.clear()
    monkeypatch.setattr(observability.logger, "propagate", True)
    caplog.set_level("INFO", logger="unified_ai.audit")
    monkeypatch.setattr(external, "_get_json", lambda _url: (_ for _ in ()).throw(OSError("offline")))

    result = call_tool("wikipedia_summary", {"topic": "Example"})
    assert result == {"error": "offline"}

    event = observability.recent_calls(1)[0]
    assert event["tool"] == "wikipedia_summary"
    assert event["status"] == "failure"
    assert event["success"] is False
    logged_event = json.loads(caplog.records[-1].getMessage())
    assert logged_event == event
    summary = observability.metrics_summary()["tools"]["wikipedia_summary"]
    assert summary["calls"] == 1
    assert summary["successes"] == 0
    assert summary["errors"] == 1
    assert summary["average_latency_ms"] >= 0


def test_tool_schemas_cover_every_handler_and_reject_extra_arguments():
    from security.validation import TOOL_INPUTS
    import server

    assert set(TOOL_INPUTS) == set(server._handlers)
    for input_model in TOOL_INPUTS.values():
        with pytest.raises(ValidationError):
            input_model.model_validate({"unexpected_argument": True})


def test_rate_limit_is_isolated_per_tool(monkeypatch):
    from security import observability

    observability._call_times["ping"].clear()
    observability._call_times["wikipedia_summary"].clear()
    monkeypatch.setenv("RATE_LIMIT_CALLS", "1")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setattr(external, "_get_json", lambda _url: {
        "title": "Example", "extract": "A short summary.",
        "content_urls": {"desktop": {"page": "https://example.test"}},
    })

    assert call_tool("ping", {}) == "pong"
    assert call_tool("wikipedia_summary", {"topic": "Example"})["title"] == "Example"
    with pytest.raises(RuntimeError, match="Rate limit"):
        call_tool("ping", {})