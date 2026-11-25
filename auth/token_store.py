"""SQLite-backed encrypted storage for provider OAuth tokens."""
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet

TokenProvider = Literal["google", "notion"]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cipher() -> Fernet:
    key = os.environ.get("TOKEN_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("Set TOKEN_ENCRYPTION_KEY before connecting OAuth providers.")
    return Fernet(key.encode("ascii"))


def _connect() -> sqlite3.Connection:
    path = Path(os.environ.get("TOKEN_DB_PATH", "var/tokens.sqlite3"))
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS oauth_tokens (provider TEXT PRIMARY KEY, token BLOB NOT NULL)"
    )
    return connection


def store_tokens(provider: TokenProvider, tokens: dict[str, Any]) -> None:
    encrypted = _cipher().encrypt(json.dumps(tokens).encode("utf-8"))
    with closing(_connect()) as connection:
        with connection:
            connection.execute(
                "INSERT INTO oauth_tokens(provider, token) VALUES (?, ?) "
                "ON CONFLICT(provider) DO UPDATE SET token = excluded.token",
                (provider, encrypted),
            )


def load_tokens(provider: TokenProvider) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT token FROM oauth_tokens WHERE provider = ?", (provider,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(_cipher().decrypt(row[0]).decode("utf-8"))