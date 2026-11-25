# MCP-Based Unified AI Tool Server

A Python 3.10+ MCP server with a Flask REST fallback and local dashboard. Every MCP and
REST tool call shares Pydantic validation, a per-tool in-process rate limit, and
structured audit logging. Fraud scoring requires a local model artifact; image
captioning uses a pretrained model downloaded on first use or supplied locally.

## Tools

| Tool | Purpose | Requirements |
|---|---|---|
| `ping` | Health check | None |
| `fraud_risk_analyzer` | XGBoost score and top SHAP contributions | `models/fraud_xgb.json` and ML extras |
| `ultracamnet_predict` | Generate a natural-language caption for an image from a local path or base64 input | ML extras; image up to 5 MB |
| `wikipedia_summary` | Wikipedia page summary | Network |
| `arxiv_search` | Search arXiv papers | Network |
| `github_repo_summary` | Public GitHub repository metadata | Network |
| `github_account_summary` | Start with a public GitHub account profile | Network |
| `github_list_repositories` | List that account's public repositories | Network |
| `github_repo_context` | Fetch selected repo metadata and README for chat follow-ups | Network |
| `weather_forecast` | Current weather from Open-Meteo | Network |
| `currency_converter` | Exchange-rate conversion | Network |
| `news_headlines` | Hacker News story search | Network |
| `google_calendar_events` | Upcoming events, read-only | Google OAuth setup |
| `notion_search` | Search pages granted to the integration | Notion OAuth setup |

The fraud training script expects CSV columns `amount`, `hour`, `foreign_country`,
`is_new_device`, and a binary target (`0` legitimate, `1` fraud). For a software
demo only, `scripts/generate_synthetic_fraud_data.py` can create reproducible
simulated training and holdout CSVs. These synthetic labels are generated from
hand-authored rules and are not real fraud data; the resulting model must not be
used to assess real transactions. Real-world evaluation requires an authorized,
representative dataset and a genuinely independent holdout. The
`ultracamnet_predict` tool now generates general image captions locally; it does
not classify histopathology images or provide medical advice.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `FLASK_SECRET_KEY` and generate a Fernet key for `TOKEN_ENCRYPTION_KEY` before
using OAuth. Generate a Fernet value with:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Put secrets only in `.env` or the deployment environment, never in source control.
Register Google and Notion OAuth applications and configure their client IDs,
secrets, and callback URLs. In Google Cloud, enable the Calendar API, create a web
application OAuth client, and register
`http://127.0.0.1:8000/auth/google/callback`. Create a Notion OAuth integration
and register `http://127.0.0.1:8000/auth/notion/callback`. Google uses
`calendar.events.readonly` for the `google_calendar_events` tool. Notion's OAuth
flow does not request a conventional scope string; `notion_search` can access
only the pages and workspace explicitly shared with that Notion integration.
Grant the integration access only to the pages it needs.

## Run

Start the MCP stdio server for Claude Desktop, VS Code, or Inspector:

```powershell
mcp dev server.py
```

Start the REST API and dashboard separately:

```powershell
python ui_server.py
```

Or run the REST API and dashboard in Docker (the image does not include model
artifacts and does not run the MCP stdio server):

```powershell
docker build -t unified-ai-tool-server .
docker run --rm -p 8000:8000 --env-file .env unified-ai-tool-server
```

The container listens on all interfaces; do not expose it publicly without an
authenticated, TLS-terminating reverse proxy. Mount `models`, `outputs`, and
`var` as volumes when persistent local model files, overlays, or OAuth token
storage are needed.

For GitHub exploration, provide a username to `github_account_summary`, pass that
username to `github_list_repositories`, then pass a selected `full_name` such as
`UmmidSalma/project-name` to `github_repo_context`. The returned public README and
metadata become context for follow-up questions in the current AI-client chat.

Open `http://127.0.0.1:8000`. REST tools are listed at `GET /api/tools` and called
with `POST /api/tools/<tool_name>` using a JSON object. Metrics and recent calls
are at `GET /api/metrics` and `GET /api/logs`. Bind to a non-loopback address only
behind an authenticated, TLS-terminating deployment; this fallback has no user
authentication.

## Models

Install optional heavyweight dependencies only when ready to use local artifacts:

```powershell
pip install -r requirements-ml.txt
python scripts/train_fraud_model.py path\to\labeled-data.csv --target label
```

Fraud inference loads `models/fraud_xgb.json`. `ultracamnet_predict` uses the
BLIP image-captioning model. Install the optional dependencies with
`.\venv\Scripts\python.exe -m pip install -r requirements-ml.txt` so they are
installed into the configured server environment. By default, the pretrained
`Salesforce/blip-image-captioning-base` weights are downloaded on first use and
cached locally; that first use needs network access. To use a checkpoint already
on disk (including offline), set `IMAGE_CAPTION_MODEL` to its local directory.
This tool is a PyTorch captioner; it does not load the separate, unfinished Keras
UltraCamNet classifier stub.
Provide a path inside the server's `IMAGE_ROOT` or base64 input, up to 5 MB.
The local dashboard also has a file picker, so an image can be uploaded directly
from your computer without copying it into the server's image folder.
Model/dependency errors are returned explicitly. Tune the initial fraud
label cutoffs against a labeled validation set using `FRAUD_MEDIUM_THRESHOLD`
and `FRAUD_HIGH_THRESHOLD`.

To chat with Claude while requiring it to call the local image tool, install the
optional SDK:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-claude.txt
```

Set `ANTHROPIC_API_KEY` in `.env`, then open `http://127.0.0.1:8000/claude`
while `python ui_server.py` is running. This local bridge sends the selected
image to both your local caption model and Anthropic's API. Do not use it for
sensitive images unless that transfer is authorized.

To exercise the fraud tool with synthetic demo data, install only its ML
dependencies and generate separate training and holdout sets:

```powershell
pip install "numpy>=1.26,<2" "pandas>=2.2,<3" "scikit-learn>=1.5,<1.8" "xgboost>=2.0,<3" "shap>=0.46,<1"
python scripts/generate_synthetic_fraud_data.py
python scripts/train_fraud_model.py data\synthetic-fraud\train.csv --target label
python scripts/evaluate.py --dataset data\synthetic-fraud\holdout.csv --target label
```

The generated CSVs and model are local, git-ignored demo artifacts. The MCP
`fraud_risk_analyzer` then loads the model from `models/fraud_xgb.json`; relative
model paths resolve from the project directory even when the MCP client starts
the server with a different working directory. Synthetic evaluation scores only
measure how well the model reproduces the generator's artificial rules.

## Tests and Evaluation

```powershell
python -m pytest tests -q
python scripts/evaluate.py --tool ping --iterations 50
```

For model metrics, pass a labeled holdout CSV using the training feature schema:

```powershell
python scripts/evaluate.py --dataset path\to\holdout.csv --target label
```

See [docs/evaluation.md](docs/evaluation.md) for metric interpretation and client
compatibility, and [docs/architecture.md](docs/architecture.md) for the diagram.
GitHub Actions runs the unit/integration suite on Python 3.10–3.12 and builds the
REST/dashboard Docker image on pushes and pull requests.

## What remains before a complete project demonstration

- Supply a legally usable, documented fraud dataset and train/holdout CSV files;
  train and report metrics on a holdout that was not used for fitting or tuning.
- Configure real Google and Notion OAuth applications, test consent and API calls,
  and keep credentials out of the repository.
- Launch Claude Desktop and VS Code Copilot with the sample configs after
  replacing the Claude absolute paths; confirm tool discovery and calls in both.
- Build and run the Docker image on a machine with Docker Desktop or another Docker
  engine. GitHub Actions also builds it on pushes and pull requests.

## Security Notes

- Image captioning accepts local images inside `IMAGE_ROOT` or base64 images up
  to 5 MB. Captions are generated locally after the pretrained model is cached.

## How to test image captioning

Open the local dashboard at `http://127.0.0.1:8000`, choose an image in
**Caption an image**, and select **Generate caption**. The image is uploaded
directly and is not saved into `IMAGE_ROOT`.

The MCP tool accepts `image_base64` or a path on the server inside `IMAGE_ROOT`.
Standard MCP tool arguments are JSON, so Claude Desktop does not automatically
forward a chat attachment to this server tool. You can attach an image in Claude
and ask Claude to describe it directly, or use the dashboard upload when you
want this server's local captioning model to process it. On first use, the
default model download requires network access.

Start MCP Inspector from the project root:

```powershell
mcp dev server.py
```

In Claude Desktop, try: **"Describe test1.png using ultracamnet_predict"**.
- OAuth tokens are encrypted at rest; protect `TOKEN_ENCRYPTION_KEY`, the token
  database, and `.env` backups.
- Audit events contain tool names, argument hashes, latency, status, and timestamp,
  not raw arguments or tokens. Events and rate limits reset on process restart.
- The dashboard is local-only by default and has no separate user authentication.
- OAuth callback state is checked. Configure exact provider callback URLs and use
  HTTPS in deployed environments.
