"""MCP-Based Unified AI Tool Server (Team 7).

Run (test in Inspector):  mcp dev server.py
Run (plain):              python server.py
"""
from mcp.server.fastmcp import FastMCP

from tools import external, fraud, ultracamnet

mcp = FastMCP("unified-ai-tool-server")


@mcp.tool()
def ping() -> str:
    """Health check. Returns 'pong' if the server is alive."""
    return "pong"


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
    return fraud.score_transaction(amount, hour, country, is_new_device)


@mcp.tool()
def ultracamnet_predict(image_path: str) -> dict:
    """Run UltraCamNet cancer detection on a histopathology image.

    Args:
        image_path: absolute path to the image file
    """
    return ultracamnet.predict_image(image_path)


@mcp.tool()
def wikipedia_summary(topic: str) -> dict:
    """Get a short Wikipedia summary for a topic."""
    return external.wikipedia_summary(topic)


@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> list | dict:
    """Search arXiv papers and return title, link, short summary."""
    return external.arxiv_search(query, max_results)


@mcp.tool()
def github_repo_summary(repo: str) -> dict:
    """Get a quick summary of a public GitHub repository."""
    return external.github_repo_summary(repo)


@mcp.tool()
def weather_forecast(city: str) -> dict:
    """Get current weather for a city using Open-Meteo."""
    return external.weather_forecast(city)


@mcp.tool()
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies using a public exchange-rate API."""
    return external.currency_converter(amount, from_currency, to_currency)


@mcp.tool()
def news_headlines(topic: str = "technology", limit: int = 5) -> list | dict:
    """Return recent tech/news headlines from Hacker News."""
    return external.news_headlines(topic, limit)


if __name__ == "__main__":
    mcp.run()
