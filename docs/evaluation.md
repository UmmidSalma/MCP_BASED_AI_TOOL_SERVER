# Evaluation

No real model or labeled dataset is included, so real-world model performance is
not claimed. To demonstrate the pipeline, generate synthetic train/holdout CSVs,
train on only the training CSV, then evaluate on the separate synthetic holdout.
These metrics measure agreement with artificial labeling rules, not fraud
detection performance. For a meaningful evaluation, supply a representative,
authorized real dataset and an independent holdout. Run:

```powershell
python scripts/generate_synthetic_fraud_data.py
python scripts/train_fraud_model.py data\synthetic-fraud\train.csv --target label
python scripts/evaluate.py --dataset data\synthetic-fraud\holdout.csv --target label --tool ping --iterations 50
```

The script evaluates every row in the supplied holdout CSV (it does not split it
again) and reports fraud accuracy, F1, ROC-AUC, and tool mean/median/p95 latency;
it writes machine-readable results to `docs/evaluation-results.json` and updates
the measured section below. Repeat latency runs on the target machine and record
whether API or model calls were included. Network-backed timings vary with providers.
The benchmark iteration count must not exceed `RATE_LIMIT_CALLS` because it
executes each sample through the normal tool dispatcher.

<!-- evaluation-results:start -->
## Latest Measurements

Run: 2026-11-12T07:44:37.954008+00:00

- Tool: `ping` (50 calls)
- Latency: mean 0.036 ms, median 0.031 ms, p95 0.053 ms
- Environment: Python 3.11.0rc2, `Windows-10-10.0.26300-SP0`
<!-- evaluation-results:end -->

## Client Compatibility Checklist

Before testing either desktop client, install the base requirements in the
workspace virtual environment and replace any sample absolute paths with the
actual project and interpreter paths. The MCP server uses stdio; do not start it
with a shell wrapper that writes status text to stdout.

| Client | Configuration | Status |
|---|---|---|
| Claude Desktop | Copy `claude_desktop_config.json` to the client MCP config location and use absolute Windows paths. | Manual client launch required |
| VS Code Copilot | Keep `.vscode/mcp.json` in the workspace; ensure its `venv` exists and has `requirements.txt` installed. | Manual client launch required |
| MCP Inspector | `mcp dev server.py` | Covered by the MCP client integration test |

For each desktop client, confirm the server starts without an error, `ping`
returns `pong`, all tools appear in the tool list, and a network-backed tool can
be called. OAuth-backed tools additionally require provider credentials and an
interactive browser consent flow.

## Project-owner inputs still required

- Provide a licensed or otherwise authorized fraud dataset with the documented
  feature columns and binary labels. Keep training and holdout files separate;
  do not fit or tune on the holdout. Record the data source, permitted use,
  preprocessing, class balance, and split strategy alongside submitted results.
- Provide and evaluate a fraud model artifact before making real-world fraud
  scoring claims. The repository's synthetic data and model are demonstration
  artifacts only.
- Image captioning uses a pretrained BLIP checkpoint downloaded on first use or
  supplied locally through `IMAGE_CAPTION_MODEL`. Verify it can load in the
  target environment and review captions for the intended use; it is not a
  medical classifier.
- Add Google/Notion application credentials in the local environment, complete
  browser consent, and record successful live API checks without publishing
  tokens or personally identifying data.
- Manually launch the configured MCP server from both Claude Desktop and VS Code
  Copilot. The automated MCP client test covers SDK discovery and `ping`, not
  those desktop applications.
- Run the Docker build and container smoke test with a working Docker engine.