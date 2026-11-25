"""Provider-specific OAuth authorization and code exchange helpers."""
import base64
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from auth.token_store import TokenProvider, load_tokens, store_tokens


PROVIDERS = {
    "google": {
        "client_id_env": "GOOGLE_CLIENT_ID",
        "client_secret_env": "GOOGLE_CLIENT_SECRET",
        "redirect_env": "GOOGLE_REDIRECT_URI",
        "redirect_default": "http://127.0.0.1:8000/auth/google/callback",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
    },
    "notion": {
        "client_id_env": "NOTION_CLIENT_ID",
        "client_secret_env": "NOTION_CLIENT_SECRET",
        "redirect_env": "NOTION_REDIRECT_URI",
        "redirect_default": "http://127.0.0.1:8000/auth/notion/callback",
        "authorize_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
    },
}


def redirect_uri(provider: TokenProvider) -> str:
    config = PROVIDERS[provider]
    return os.environ.get(config["redirect_env"], config["redirect_default"])


def authorization_url(provider: TokenProvider, state: str) -> str:
    config = PROVIDERS[provider]
    client_id = os.environ.get(config["client_id_env"])
    if not client_id or not os.environ.get(config["client_secret_env"]):
        raise RuntimeError(f"OAuth client credentials for {provider} are not configured.")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(provider),
        "response_type": "code",
        "state": state,
    }
    if provider == "google":
        params.update({
            "scope": "https://www.googleapis.com/auth/calendar.events.readonly",
            "access_type": "offline",
            "prompt": "consent",
        })
    else:
        params["owner"] = "user"
    return config["authorize_url"] + "?" + urllib.parse.urlencode(params)


def _post_form(url: str, fields: dict[str, str], provider: TokenProvider) -> dict[str, Any]:
    config = PROVIDERS[provider]
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if provider == "notion":
        credentials = f"{os.environ[config['client_id_env']]}:{os.environ[config['client_secret_env']]}"
        headers["Authorization"] = "Basic " + base64.b64encode(credentials.encode()).decode()
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode(), headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def exchange_code(provider: TokenProvider, code: str) -> None:
    config = PROVIDERS[provider]
    client_id = os.environ[config["client_id_env"]]
    client_secret = os.environ[config["client_secret_env"]]
    fields = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri(provider),
    }
    if provider == "google":
        fields.update({"client_id": client_id, "client_secret": client_secret})
    tokens = _post_form(config["token_url"], fields, provider)
    if tokens.get("expires_in") is not None:
        tokens["expires_at"] = time.time() + int(tokens["expires_in"])
    store_tokens(provider, tokens)


def access_token(provider: TokenProvider) -> str:
    tokens = load_tokens(provider)
    if not tokens:
        raise RuntimeError(f"Connect {provider.title()} first using the dashboard OAuth link.")
    if tokens.get("expires_at") and tokens["expires_at"] <= time.time() + 60:
        refresh_token = tokens.get("refresh_token")
        if provider != "google" or not refresh_token:
            raise RuntimeError(f"The {provider.title()} token expired; reconnect the account.")
        config = PROVIDERS[provider]
        refreshed = _post_form(config["token_url"], {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": os.environ[config["client_id_env"]],
            "client_secret": os.environ[config["client_secret_env"]],
        }, provider)
        refreshed.setdefault("refresh_token", refresh_token)
        refreshed["expires_at"] = time.time() + int(refreshed.get("expires_in", 3600))
        store_tokens(provider, refreshed)
        tokens = refreshed
    token = tokens.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"No valid access token is stored for {provider.title()}; reconnect the account.")
    return token