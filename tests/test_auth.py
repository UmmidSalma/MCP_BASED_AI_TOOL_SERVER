import pytest
from urllib.parse import parse_qs, urlparse

from cryptography.fernet import Fernet

from auth import oauth, token_store
from auth.oauth import access_token, authorization_url


def test_oauth_token_storage_is_encrypted(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    database = tmp_path / "tokens.sqlite3"
    monkeypatch.setenv("TOKEN_DB_PATH", str(database))
    token_store.store_tokens("google", {"access_token": "secret-token"})
    assert token_store.load_tokens("google") == {"access_token": "secret-token"}
    assert b"secret-token" not in database.read_bytes()


def test_relative_token_database_path_uses_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    working_directory = tmp_path / "working"
    project_root.mkdir()
    working_directory.mkdir()
    monkeypatch.setattr(token_store, "_PROJECT_ROOT", project_root)
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("TOKEN_DB_PATH", "var/tokens.sqlite3")

    token_store.store_tokens("google", {"access_token": "secret-token"})

    assert (project_root / "var" / "tokens.sqlite3").is_file()
    assert not (working_directory / "var").exists()


def test_google_oauth_requests_read_only_calendar_scope(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    url = authorization_url("google", "random-state")
    params = parse_qs(urlparse(url).query)
    assert params["scope"] == ["https://www.googleapis.com/auth/calendar.events.readonly"]
    assert params["state"] == ["random-state"]


def test_notion_oauth_uses_integration_page_permissions(monkeypatch):
    monkeypatch.setenv("NOTION_CLIENT_ID", "client-id")
    monkeypatch.setenv("NOTION_CLIENT_SECRET", "client-secret")
    params = parse_qs(urlparse(authorization_url("notion", "random-state")).query)
    assert params["owner"] == ["user"]
    assert params["state"] == ["random-state"]
    assert "scope" not in params


def test_access_token_rejects_malformed_stored_token(monkeypatch):
    monkeypatch.setattr(oauth, "load_tokens", lambda provider: {"provider": provider})
    with pytest.raises(RuntimeError, match="No valid access token"):
        access_token("google")


def test_oauth_callback_requires_session_state():
    from ui_server import app

    app.config["TESTING"] = True
    app.secret_key = "test-session-secret"
    client = app.test_client()
    with client.session_transaction() as session:
        session["oauth_state_google"] = "expected-state"
    response = client.get("/auth/google/callback?code=auth-code&state=wrong-state")
    assert response.status_code == 400