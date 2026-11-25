"""MCP entry point for the Slice 1 UltraCamNet stub."""
from mcp.server.fastmcp import FastMCP

from tools.ultracamnet import predict_image


mcp = FastMCP("unified-ai-tools")


@mcp.tool()
def ultracamnet_predict(image_base64: str) -> dict[str, str | float]:
    """Placeholder image-classification tool; the model is not connected yet."""
    return predict_image(image_base64)


if __name__ == "__main__":
    mcp.run()
