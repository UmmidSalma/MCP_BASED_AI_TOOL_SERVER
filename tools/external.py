"""External tools: Wikipedia, arXiv, GitHub, and weather (no OAuth needed)."""
import base64
import certifi
import json
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "mcp-unified-tool-server/0.1"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=15, context=context) as r:
        return r.read()


def _get_json(url: str) -> dict | list:
    return json.loads(_get(url).decode("utf-8"))


def wikipedia_summary(topic: str) -> dict:
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.parse.quote(topic.replace(" ", "_")))
    try:
        data = _get_json(url)
        return {"title": data.get("title"), "summary": data.get("extract"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page")}
    except Exception as e:
        return {"error": str(e)}


def arxiv_search(query: str, max_results: int = 3) -> list | dict:
    url = ("https://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(query) + f"&max_results={max_results}")
    try:
        root = ET.fromstring(_get(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in root.findall("a:entry", ns):
            out.append({
                "title": " ".join(e.find("a:title", ns).text.split()),
                "link": e.find("a:id", ns).text,
                "summary": " ".join(e.find("a:summary", ns).text.split())[:300],
            })
        return out
    except Exception as e:
        return {"error": str(e)}


def github_repo_summary(repo: str) -> dict:
    repo = repo.strip().strip("/")
    if not _valid_github_repo(repo):
        return {"error": "Please provide a repository in 'owner/name' format."}
    url = f"https://api.github.com/repos/{urllib.parse.quote(repo, safe='/')}"
    try:
        data = _get_json(url)
        return _repo_fields(data)
    except Exception as e:
        return {"error": str(e)}


def _valid_github_repo(repo: str) -> bool:
    return re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", repo,
    ) is not None


def _valid_github_username(username: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", username) is not None


def _repo_fields(data: dict) -> dict:
    return {
        "name": data.get("full_name"),
        "description": data.get("description"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "language": data.get("language"),
        "open_issues": data.get("open_issues_count"),
        "default_branch": data.get("default_branch"),
        "topics": data.get("topics", []),
        "license": (data.get("license") or {}).get("spdx_id"),
        "updated_at": data.get("updated_at"),
        "url": data.get("html_url"),
    }


def github_account_summary(username: str) -> dict:
    """Return public profile details to begin an account-first GitHub flow."""
    username = username.strip().removeprefix("@").strip()
    if not _valid_github_username(username):
        return {"error": "Enter a GitHub username, not a profile URL."}
    try:
        data = _get_json(f"https://api.github.com/users/{urllib.parse.quote(username)}")
        return {
            "username": data.get("login"),
            "name": data.get("name"),
            "bio": data.get("bio"),
            "company": data.get("company"),
            "location": data.get("location"),
            "public_repositories": data.get("public_repos"),
            "followers": data.get("followers"),
            "following": data.get("following"),
            "profile_url": data.get("html_url"),
            "avatar_url": data.get("avatar_url"),
            "created_at": data.get("created_at"),
        }
    except Exception as e:
        return {"error": str(e)}


def github_list_repositories(username: str, limit: int = 20) -> dict:
    """List public repositories for an account, newest activity first."""
    username = username.strip().removeprefix("@").strip()
    if not _valid_github_username(username):
        return {"error": "Enter a GitHub username, not a profile URL."}
    limit = max(1, min(int(limit), 100))
    query = urllib.parse.urlencode({"sort": "updated", "direction": "desc", "per_page": limit})
    try:
        data = _get_json(f"https://api.github.com/users/{urllib.parse.quote(username)}/repos?{query}")
        repositories = [
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "description": item.get("description"),
                "language": item.get("language"),
                "stars": item.get("stargazers_count"),
                "forks": item.get("forks_count"),
                "updated_at": item.get("updated_at"),
                "is_fork": item.get("fork", False),
                "archived": item.get("archived", False),
                "url": item.get("html_url"),
            }
            for item in data
        ]
        return {"account": username, "count": len(repositories), "repositories": repositories}
    except Exception as e:
        return {"error": str(e)}


def github_repo_context(repo: str) -> dict:
    """Fetch public repository metadata and a bounded README for chat follow-ups."""
    repo = repo.strip().strip("/")
    if not _valid_github_repo(repo):
        return {"error": "Please select a repository in 'owner/name' format."}
    api_repo = urllib.parse.quote(repo, safe="/")
    try:
        metadata = _get_json(f"https://api.github.com/repos/{api_repo}")
        readme = _get_json(f"https://api.github.com/repos/{api_repo}/readme")
        content = base64.b64decode(readme.get("content", "")).decode("utf-8", errors="replace")
        excerpt_limit = 12000
        return {
            "repository": _repo_fields(metadata),
            "readme": content[:excerpt_limit],
            "readme_truncated": len(content) > excerpt_limit,
            "conversation_hint": "Use this public repository context to answer follow-up questions in the current chat.",
        }
    except Exception as e:
        return {"error": str(e)}


def weather_forecast(city: str) -> dict:
    query = urllib.parse.quote(city)
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={query}&count=1&language=en&format=json"
    )
    try:
        geo = _get_json(geo_url)
        if not geo.get("results"):
            return {"error": f"No location found for '{city}'."}
        loc = geo["results"][0]
        lat = loc["latitude"]
        lon = loc["longitude"]
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_2m,weather_code&timezone=auto"
        )
        data = _get_json(weather_url)
        current = data.get("current", {})
        return {
            "city": loc.get("name"),
            "country": loc.get("country"),
            "latitude": lat,
            "longitude": lon,
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_kph": current.get("wind_speed_2m"),
            "weather_code": current.get("weather_code"),
            "timezone": data.get("timezone"),
        }
    except Exception as e:
        return {"error": str(e)}


def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    from_currency = from_currency.strip().upper()
    to_currency = to_currency.strip().upper()
    if not from_currency or not to_currency:
        return {"error": "Please provide both from_currency and to_currency."}
    try:
        data = _get_json(f"https://open.er-api.com/v6/latest/{from_currency}")
        rates = data.get("rates", {})
        if not rates or to_currency not in rates:
            return {"error": f"Unsupported conversion: {from_currency} -> {to_currency}"}
        converted = amount * rates[to_currency]
        return {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rates[to_currency],
            "converted_amount": round(converted, 4),
            "source": data.get("provider"),
        }
    except Exception as e:
        return {"error": str(e)}


def news_headlines(topic: str = "technology", limit: int = 5) -> list | dict:
    topic = (topic or "technology").strip() or "technology"
    limit = max(1, min(int(limit or 5), 10))
    try:
        url = (
            "https://hn.algolia.com/api/v1/search_by_date?"
            f"query={urllib.parse.quote(topic)}&tags=story&hitsPerPage={limit}"
        )
        data = _get_json(url)
        hits = data.get("hits", [])
        out = []
        for item in hits:
            out.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "author": item.get("author"),
                "points": item.get("points"),
                "created_at": item.get("created_at"),
            })
        return out
    except Exception as e:
        return {"error": str(e)}
