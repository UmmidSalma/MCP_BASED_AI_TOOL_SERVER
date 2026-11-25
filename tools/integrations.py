"""OAuth-backed Google Calendar and Notion tools."""
import json
from datetime import datetime, timezone
import urllib.parse
import urllib.request
from typing import Any

from auth.oauth import access_token


def google_calendar_events(max_results: int = 10) -> dict[str, Any]:
    """Fetch upcoming events using the read-only Calendar events scope."""
    query = urllib.parse.urlencode({
        "maxResults": max_results,
        "orderBy": "startTime",
        "singleEvents": "true",
        "timeMin": datetime.now(timezone.utc).isoformat(),
    })
    request = urllib.request.Request(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events?" + query,
        headers={"Authorization": "Bearer " + access_token("google")},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {"events": data.get("items", []), "count": len(data.get("items", []))}


def notion_search(query: str) -> dict[str, Any]:
    """Search pages visible to the authorized Notion connection."""
    request = urllib.request.Request(
        "https://api.notion.com/v1/search",
        data=json.dumps({"query": query, "page_size": 10}).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + access_token("notion"),
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    results = data.get("results", [])
    return {"results": results, "count": len(results)}