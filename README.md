# MCP-Based Unified AI Tool Server (Team 7)

One MCP server exposing ML models and external tools through a standard interface.

## Tools
| Tool | What it does |
|---|---|
| ping | health check |
| fraud_risk_analyzer | fraud risk score for a payment |
| ultracamnet_predict | cancer detection on an image |
| wikipedia_summary | Wikipedia summary |
| arxiv_search | arXiv paper search |

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## UI Dashboard
Launch the interactive dashboard for a cleaner visual interface:
```
python ui_server.py
```
Then open `http://127.0.0.1:8000` in your browser.

## Test
```
mcp dev server.py
```
Opens MCP Inspector. Click Tools, pick one, Run.

## Claude Desktop
Add to `claude_desktop_config.json` (Settings > Developer > Edit Config):
```json
{
  "mcpServers": {
    "unified-ai": {
      "command": "C:/path/to/venv/Scripts/python.exe",
      "args": ["C:/path/to/MCP-Based-Unified-AI-Tool-Server/server.py"]
    }
  }
}
```
Restart Claude Desktop.

## Status
- [x] Protocol working (demo tools)
- [ ] Real Fraud model
- [ ] Real UltraCamNet model + Grad-CAM
- [ ] OAuth tools (GitHub, Calendar, Notion)
