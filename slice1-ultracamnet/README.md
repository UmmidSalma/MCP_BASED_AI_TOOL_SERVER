# UltraCamNet MCP Starter — Slice 1

This isolated starter exposes one MCP tool, `ultracamnet_predict`, through
stdio. It currently returns a placeholder; image input and the model are not
connected yet. Every response includes a research-only disclaimer.

The existing project server is outside this folder and is unchanged. The model
file `models/ultracamnet.keras` is not present in the workspace; do not put real
patient images into tests.

## Setup

From this folder, create and activate a virtual environment, then install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Run in MCP Inspector

From this folder, start the server with:

```powershell
mcp dev server.py
```

In MCP Inspector, confirm that `ultracamnet_predict` is listed. Call it with
any string for `image_base64`; this stub does not decode or inspect the input.
The response should contain `prediction: "stub"` and the research disclaimer.

## Run the test

```powershell
pytest -q
```
