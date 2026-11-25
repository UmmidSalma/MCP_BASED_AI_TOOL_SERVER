"""Input validation and image-file access policy for public tool calls."""
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


ShortText = Annotated[str, Field(min_length=1, max_length=200)]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PingInput(ToolInput):
    pass


class FraudInput(ToolInput):
    amount: float = Field(gt=0, le=100_000_000, allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    country: ShortText
    is_new_device: bool = False


class ImageInput(ToolInput):
    image_path: str | None = None
    image_base64: str | None = None


class WikipediaInput(ToolInput):
    topic: ShortText


class ArxivInput(ToolInput):
    query: ShortText
    max_results: int = Field(default=3, ge=1, le=10)


class GitHubInput(ToolInput):
    repo: Annotated[str, Field(
        min_length=3,
        max_length=201,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$",
    )]


class GitHubAccountInput(ToolInput):
    username: Annotated[str, Field(
        min_length=1,
        max_length=40,
        pattern=r"^@?[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$",
    )]


class GitHubRepositoriesInput(GitHubAccountInput):
    limit: int = Field(default=20, ge=1, le=100)


class WeatherInput(ToolInput):
    city: ShortText


class CurrencyInput(ToolInput):
    amount: float = Field(ge=0, le=1_000_000_000, allow_inf_nan=False)
    from_currency: Annotated[str, Field(min_length=3, max_length=3)]
    to_currency: Annotated[str, Field(min_length=3, max_length=3)]


class NewsInput(ToolInput):
    topic: Annotated[str, Field(default="technology", max_length=200)] = "technology"
    limit: int = Field(default=5, ge=1, le=10)


class CalendarInput(ToolInput):
    max_results: int = Field(default=10, ge=1, le=100)


class NotionInput(ToolInput):
    query: Annotated[str, Field(min_length=1, max_length=200)]


TOOL_INPUTS: dict[str, type[ToolInput]] = {
    "ping": PingInput,
    "fraud_risk_analyzer": FraudInput,
    "ultracamnet_predict": ImageInput,
    "wikipedia_summary": WikipediaInput,
    "arxiv_search": ArxivInput,
    "github_repo_summary": GitHubInput,
    "github_account_summary": GitHubAccountInput,
    "github_list_repositories": GitHubRepositoriesInput,
    "github_repo_context": GitHubInput,
    "weather_forecast": WeatherInput,
    "currency_converter": CurrencyInput,
    "news_headlines": NewsInput,
    "google_calendar_events": CalendarInput,
    "notion_search": NotionInput,
}


def validate_image_file(
    image_path: str,
    max_bytes: int = 5 * 1024 * 1024,
) -> Path:
    """Resolve an allowed image inside IMAGE_ROOT and enforce its size limit."""
    root = Path(os.environ.get("IMAGE_ROOT", "images"))
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    root = root.resolve()
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Image path must be inside the configured IMAGE_ROOT.") from exc
    try:
        resolved = resolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Image file does not exist.") from exc

    if not resolved.is_file():
        raise ValueError("Image path must refer to a regular file.")
    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise ValueError("Unsupported image type; use JPEG, PNG, BMP, or TIFF.")
    if resolved.stat().st_size > max_bytes:
        raise ValueError("Image exceeds the 5 MB size limit.")
    return resolved