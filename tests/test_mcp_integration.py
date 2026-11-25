import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from server import _handlers

def test_mcp_lists_and_calls_tools():
    async def run_client():
        project_root = Path(__file__).resolve().parents[1]
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(project_root / "server.py")],
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                assert names == set(_handlers)
                result = await client.call_tool("ping", {})
                assert result.content[0].text == "pong"
                image_result = await client.call_tool(
                    "ultracamnet_predict", {"image_base64": "not-base64"}
                )
                assert "valid standard base64" in image_result.content[0].text

    asyncio.run(run_client())