"""MCP-Based Unified AI Tool Server (Team 7).

Run (test in Inspector):  mcp dev server.py
Run (plain):              python server.py
"""
import asyncio

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

from security.observability import execute_async_tool, execute_tool
from tools import external, fraud, integrations, ultracamnet

load_dotenv()

mcp = FastMCP("unified-ai-tool-server")

_handlers = {
    "ping": lambda: "pong",
    "fraud_risk_analyzer": fraud.score_transaction,
    "ultracamnet_predict": ultracamnet.predict_image,
    "wikipedia_summary": external.wikipedia_summary,
    "arxiv_search": external.arxiv_search,
    "github_repo_summary": external.github_repo_summary,
    "github_account_summary": external.github_account_summary,
    "github_list_repositories": external.github_list_repositories,
    "github_repo_context": external.github_repo_context,
    "weather_forecast": external.weather_forecast,
    "currency_converter": external.currency_converter,
    "news_headlines": external.news_headlines,
    "google_calendar_events": integrations.google_calendar_events,
    "notion_search": integrations.notion_search,
}


def call_tool(tool_name: str, arguments: dict) -> object:
    """Invoke a public tool through shared validation and instrumentation."""
    if tool_name == "ultracamnet_predict":
        return asyncio.run(execute_async_tool(tool_name, arguments, _handlers[tool_name]))
    return execute_tool(tool_name, arguments, _handlers[tool_name])


@mcp.tool()
def ping() -> str:
    """Health check. Returns 'pong' if the server is alive."""
    return call_tool("ping", {})


@mcp.tool()
def fraud_risk_analyzer(amount: float, hour: int, country: str,
                        is_new_device: bool = False) -> dict:
    """Score a digital payment for fraud risk.

    Args:
        amount: transaction amount in INR
        hour: hour of day, 0-23
        country: country code or name, e.g. "IN"
        is_new_device: True if from an unrecognised device
    """
    return call_tool("fraud_risk_analyzer", {
        "amount": amount, "hour": hour, "country": country,
        "is_new_device": is_new_device,
    })


def ultracamnet_predict(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict:
    """Run image captioning for direct, non-event-loop callers."""
    return asyncio.run(_execute_ultracamnet_predict(image_path, image_base64))


async def _execute_ultracamnet_predict(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict:
    return await execute_async_tool(
        "ultracamnet_predict",
        {"image_path": image_path, "image_base64": image_base64},
        ultracamnet.predict_image,
    )


@mcp.tool(name="ultracamnet_predict")
async def ultracamnet_predict_tool(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict:
    """Caption an image using server-local IMAGE_ROOT or client-supplied base64.

    Args:
        image_path: image path inside the server's IMAGE_ROOT
        image_base64: base64-encoded image file, up to 5 MB when decoded
    """
    return await _execute_ultracamnet_predict(image_path, image_base64)


@mcp.tool()
def wikipedia_summary(topic: str) -> dict:
    """Get a short Wikipedia summary for a topic."""
    return call_tool("wikipedia_summary", {"topic": topic})


@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> list | dict:
    """Search arXiv papers and return title, link, short summary."""
    return call_tool("arxiv_search", {"query": query, "max_results": max_results})


@mcp.tool()
def github_repo_summary(repo: str) -> dict:
    """Get a quick metadata summary for one public GitHub repository."""
    return call_tool("github_repo_summary", {"repo": repo})


@mcp.tool()
def github_account_summary(username: str) -> dict:
    """Start the GitHub flow: ask for a username if missing, then summarize its public account."""
    return call_tool("github_account_summary", {"username": username})


@mcp.tool()
def github_list_repositories(username: str, limit: int = 20) -> dict:
    """After account lookup, list public repositories and ask which full_name to explore."""
    return call_tool("github_list_repositories", {"username": username, "limit": limit})


@mcp.tool()
def github_repo_context(repo: str) -> dict:
    """Load the selected public repository's metadata and README for continued chat follow-ups."""
    return call_tool("github_repo_context", {"repo": repo})


@mcp.tool()
def weather_forecast(city: str) -> dict:
    """Get current weather for a city using Open-Meteo."""
    return call_tool("weather_forecast", {"city": city})


@mcp.tool()
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies using a public exchange-rate API."""
    return call_tool("currency_converter", {
        "amount": amount, "from_currency": from_currency, "to_currency": to_currency,
    })


@mcp.tool()
def news_headlines(topic: str = "technology", limit: int = 5) -> list | dict:
    """Return recent tech/news headlines from Hacker News."""
    return call_tool("news_headlines", {"topic": topic, "limit": limit})


@mcp.tool()
def google_calendar_events(max_results: int = 10) -> dict:
    """List upcoming Google Calendar events (read-only OAuth scope)."""
    return call_tool("google_calendar_events", {"max_results": max_results})


@mcp.tool()
def notion_search(query: str) -> dict:
    """Search pages available to the connected Notion workspace."""
    return call_tool("notion_search", {"query": query})


if __name__ == "__main__":
    mcp.run()
