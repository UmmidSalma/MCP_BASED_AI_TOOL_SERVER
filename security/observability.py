"""Per-process tool-call throttling, audit logging, and metrics."""
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from security.validation import TOOL_INPUTS


logger = logging.getLogger("unified_ai.audit")
logger.propagate = False
logger.setLevel(logging.INFO)
if not logger.handlers:
    audit_handler = logging.StreamHandler(sys.stderr)
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(audit_handler)
_lock = threading.Lock()
_call_times: dict[str, deque[float]] = defaultdict(deque)
_events: deque[dict[str, Any]] = deque(maxlen=500)


def _check_rate_limit(tool_name: str) -> None:
    limit = max(1, int(os.environ.get("RATE_LIMIT_CALLS", "60")))
    window = max(1, int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")))
    now = time.monotonic()
    with _lock:
        calls = _call_times[tool_name]
        while calls and now - calls[0] >= window:
            calls.popleft()
        if len(calls) >= limit:
            raise RuntimeError(f"Rate limit exceeded for {tool_name}.")
        calls.append(now)


def execute_tool(tool_name: str, arguments: dict[str, Any], handler: Callable[..., Any]) -> Any:
    """Validate, throttle, invoke, and record one tool call."""
    started = time.perf_counter()
    args_hash = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    status = "failure"
    try:
        input_model = TOOL_INPUTS[tool_name]
        validated = input_model.model_validate(arguments)
        _check_rate_limit(tool_name)
        result = handler(**validated.model_dump())
        status = "failure" if isinstance(result, dict) and result.get("error") else "success"
        return result
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        event = {
            "tool": tool_name,
            "args_hash": args_hash,
            "latency_ms": duration_ms,
            "status": status,
            "success": status == "success",
            "timestamp": time.time(),
        }
        with _lock:
            _events.append(event)
        logger.info(json.dumps(event, separators=(",", ":")))


async def execute_async_tool(
    tool_name: str,
    arguments: dict[str, Any],
    handler: Callable[..., Awaitable[Any]],
) -> Any:
    """Validate, throttle, await, and record one asynchronous tool call."""
    started = time.perf_counter()
    args_hash = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    status = "failure"
    try:
        input_model = TOOL_INPUTS[tool_name]
        validated = input_model.model_validate(arguments)
        _check_rate_limit(tool_name)
        result = await handler(**validated.model_dump())
        status = "failure" if isinstance(result, dict) and result.get("error") else "success"
        return result
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        event = {
            "tool": tool_name,
            "args_hash": args_hash,
            "latency_ms": duration_ms,
            "status": status,
            "success": status == "success",
            "timestamp": time.time(),
        }
        with _lock:
            _events.append(event)
        logger.info(json.dumps(event, separators=(",", ":")))


def recent_calls(limit: int = 100) -> list[dict[str, Any]]:
    with _lock:
        count = max(0, min(limit, 500))
        if count == 0:
            return []
        return [dict(event) for event in list(_events)[-count:]][::-1]


def metrics_summary() -> dict[str, Any]:
    with _lock:
        events = list(_events)
    by_tool: dict[str, dict[str, Any]] = {}
    for event in events:
        tool = event["tool"]
        summary = by_tool.setdefault(tool, {
            "calls": 0,
            "successes": 0,
            "errors": 0,
            "latency_ms_total": 0.0,
            "max_latency_ms": 0.0,
        })
        summary["calls"] += 1
        succeeded = event["status"] == "success"
        summary["successes"] += succeeded
        summary["errors"] += not succeeded
        summary["latency_ms_total"] += event["latency_ms"]
        summary["max_latency_ms"] = max(summary["max_latency_ms"], event["latency_ms"])
    for summary in by_tool.values():
        calls = summary["calls"]
        summary["average_latency_ms"] = round(summary.pop("latency_ms_total") / calls, 3)
    return {"calls_recorded": len(events), "tools": by_tool}