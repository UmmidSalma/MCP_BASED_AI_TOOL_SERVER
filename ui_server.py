"""Flask REST fallback and local tool-call dashboard."""
import base64
import io
import os
import secrets

from flask import Flask, jsonify, redirect, render_template_string, request, session
from werkzeug.exceptions import RequestEntityTooLarge
from pydantic import ValidationError

from auth.oauth import authorization_url, exchange_code
from security.observability import metrics_summary, recent_calls
from security.validation import TOOL_INPUTS
from server import call_tool
from tools.claude_image_chat import caption_with_claude


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=_MAX_IMAGE_BYTES + 64 * 1024,
)
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"

_DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unified AI Tool Server</title><style>
body{font:15px/1.5 Georgia,serif;margin:0;background:#f2f4ef;color:#1b2822}main{max-width:1000px;margin:36px auto;padding:0 20px}
h1{font-size:30px;margin:0 0 4px}p{color:#53645a;margin:0 0 24px}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
section{padding:16px 0;border-top:1px solid #bdc9c0}h2{font:600 16px Arial,sans-serif;margin:0 0 8px}a{color:#12654b;margin-right:16px}
table{width:100%;border-collapse:collapse;text-align:left;background:#e6ebe5}th,td{padding:9px 10px;border-bottom:1px solid #cbd4cc}th{font:600 12px Arial,sans-serif;text-transform:uppercase;color:#53645a}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}.card{background:#e6ebe5;padding:12px}.card strong{display:block;font:600 20px Arial,sans-serif}
.scroll{overflow:auto;max-height:420px}.muted{color:#64736a}.failure{color:#a12727}.success{color:#12654b}
</style></head><body><main><h1>Unified AI Tool Server</h1><p>Local status, OAuth connections, and recent tool activity.</p>
<section><h2>Connections</h2><a href="/auth/google/start">Connect Google Calendar</a><a href="/auth/notion/start">Connect Notion</a></section>
<section><h2>Claude + Ultracam</h2><a href="/claude">Open image chat</a><p class="muted">Claude will call the local ultracamnet_predict tool for uploaded images.</p></section>
<section><h2>Caption an image</h2><form id="image-form"><label>Choose an image <input name="image" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff" required></label> <button type="submit">Generate caption</button></form><p id="image-result" class="muted" aria-live="polite"></p></section>
<section><h2>Metrics</h2><div id="metrics" class="cards"><span class="muted">Loading...</span></div></section>
<div class="row"><section><h2>Per-tool summary</h2><div class="scroll"><table><thead><tr><th>Tool</th><th>Calls</th><th>Success</th><th>Failures</th><th>Avg ms</th><th>Max ms</th></tr></thead><tbody id="tool-metrics"></tbody></table></div></section>
<section><h2>Registered tools</h2><div id="tools" class="muted">Loading...</div></section></div>
<section><h2>Recent calls</h2><div class="scroll"><table><thead><tr><th>Time</th><th>Tool</th><th>Result</th><th>Latency</th><th>Argument hash</th></tr></thead><tbody id="calls"></tbody></table></div></section></main>
<script>
const cell=(row,value,className='')=>{const td=document.createElement('td');td.textContent=value;td.className=className;row.appendChild(td)};
async function load(){try{
 const [metricsResponse,toolsResponse,callsResponse]=await Promise.all([fetch('/api/metrics'),fetch('/api/tools'),fetch('/api/logs')]);
 if(!metricsResponse.ok||!toolsResponse.ok||!callsResponse.ok)throw new Error('Dashboard API request failed');
 const metrics=await metricsResponse.json(),tools=await toolsResponse.json(),calls=await callsResponse.json();
 const totals=Object.values(metrics.tools).reduce((a,t)=>({calls:a.calls+t.calls,successes:a.successes+t.successes,failures:a.failures+t.errors}),{calls:0,successes:0,failures:0});
 const cards=document.getElementById('metrics');cards.replaceChildren();
 for(const [label,value] of [['Recorded calls',metrics.calls_recorded],['Successes',totals.successes],['Failures',totals.failures]]){const card=document.createElement('div');card.className='card';const strong=document.createElement('strong');strong.textContent=value;card.append(strong,document.createTextNode(label));cards.appendChild(card)}
 const metricRows=document.getElementById('tool-metrics');metricRows.replaceChildren();
 for(const [name,item] of Object.entries(metrics.tools)){const row=document.createElement('tr');[name,item.calls,item.successes,item.errors,item.average_latency_ms,item.max_latency_ms].forEach(value=>cell(row,String(value)));metricRows.appendChild(row)}
 const toolList=document.getElementById('tools');toolList.textContent=tools.tools.join(', ');
 const callRows=document.getElementById('calls');callRows.replaceChildren();
 for(const item of calls.calls){const row=document.createElement('tr');cell(row,new Date(item.timestamp*1000).toLocaleTimeString());cell(row,item.tool);cell(row,item.status,item.success?'success':'failure');cell(row,`${item.latency_ms} ms`);cell(row,item.args_hash.slice(0,12));callRows.appendChild(row)}
 }catch(error){document.getElementById('metrics').textContent=error.message}}
load();setInterval(load,5000)
</script>
<script>
document.getElementById('image-form').addEventListener('submit',async event=>{
 event.preventDefault();
 const output=document.getElementById('image-result');
 output.textContent='Generating caption...';
 try{
  const response=await fetch('/api/tools/ultracamnet_predict',{method:'POST',body:new FormData(event.currentTarget)});
  const data=await response.json();
  const result=data.result||data;
  output.textContent=result.caption||result.error||data.error||'Caption request failed.';
 }catch(error){output.textContent=`Upload failed: ${error.message}`}
});
</script>
</body></html>"""

_CLAUDE_CHAT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude + Ultracam</title><style>
body{font:16px/1.5 Arial,sans-serif;margin:0;background:#f2f4ef;color:#1b2822}main{max-width:760px;margin:40px auto;padding:0 20px}
section{padding:18px 0;border-top:1px solid #bdc9c0}label{display:block;margin:10px 0}textarea{width:100%;min-height:90px}
button{padding:8px 14px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#53645a}
</style></head><body><main><h1>Claude + Ultracam image chat</h1>
<p class="muted">The uploaded image is sent to Claude and to your local ultracamnet_predict model. Maximum image size: 5 MB.</p>
<p><a href="/">Back to dashboard</a></p>
<section><form id="claude-form">
<label>Image (JPEG or PNG) <input name="image" type="file" accept="image/jpeg,image/png" required></label>
<label>Message <textarea name="message">Use ultracamnet_predict to caption this image, then explain the caption briefly.</textarea></label>
<button type="submit">Ask Claude with Ultracam</button>
</form></section>
<section><h2>Ultracam caption</h2><pre id="caption" aria-live="polite"></pre>
<h2>Claude response</h2><pre id="reply" aria-live="polite"></pre></section></main>
<script>
document.getElementById('claude-form').addEventListener('submit',async event=>{
 event.preventDefault();
 const caption=document.getElementById('caption'),reply=document.getElementById('reply');
 caption.textContent='Calling ultracamnet_predict...';reply.textContent='';
 try{
  const response=await fetch('/api/claude-image-chat',{method:'POST',body:new FormData(event.currentTarget)});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Image chat failed.');
  caption.textContent=data.result.caption;
  reply.textContent=data.result.reply;
 }catch(error){caption.textContent=error.message}
});
</script></body></html>"""


@app.get("/")
def dashboard():
    return render_template_string(_DASHBOARD)


@app.get("/claude")
def claude_image_chat_page():
    return render_template_string(_CLAUDE_CHAT)


@app.get("/api/tools")
def list_tools():
    return jsonify({"tools": list(TOOL_INPUTS)})


@app.post("/api/claude-image-chat")
def invoke_claude_image_chat():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a JPEG or PNG image to upload."}), 400
    image_bytes = uploaded.stream.read(_MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        return jsonify({"error": "Image exceeds the 5 MB size limit."}), 400
    prompt = request.form.get("message", "").strip()
    if not prompt or len(prompt) > 2000:
        return jsonify({"error": "Message must contain 1 to 2000 characters."}), 400
    try:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise RuntimeError(
                "Claude image chat needs Pillow; install it with "
                "`python -m pip install -r requirements-ml.txt`."
            ) from exc
        with Image.open(io.BytesIO(image_bytes)) as image:
            media_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
            }.get(image.format)
            image.verify()
        if media_type is None:
            return jsonify({"error": "Claude image chat accepts JPEG or PNG images."}), 400
        result = caption_with_claude(image_bytes, media_type, prompt, call_tool)
        return jsonify({"result": result})
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a supported image."}), 400
    except OSError:
        return jsonify({"error": "The uploaded file is not a supported image."}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Claude image chat failed")
        return jsonify({"error": "Claude image chat failed; check server logs."}), 502


@app.post("/api/tools/<tool_name>")
def invoke_tool(tool_name: str):
    if tool_name not in TOOL_INPUTS:
        return jsonify({"error": "Unknown tool", "tool": tool_name}), 404
    if tool_name == "ultracamnet_predict" and "image" in request.files:
        image_bytes = request.files["image"].stream.read(_MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            return jsonify({"error": "Image exceeds the 5 MB size limit."}), 400
        arguments = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii")
        }
    else:
        arguments = request.get_json(silent=True)
    if not isinstance(arguments, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        return jsonify({"result": call_tool(tool_name, arguments)})
    except ValidationError as exc:
        return jsonify({"error": "Invalid tool arguments", "details": exc.errors(include_url=False)}), 400
    except RuntimeError as exc:
        status = 429 if "Rate limit" in str(exc) else 503
        return jsonify({"error": str(exc)}), status
    except FileNotFoundError:
        return jsonify({"error": "A required local model artifact is unavailable."}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Tool call failed: %s", tool_name)
        return jsonify({"error": "Tool call failed; see server audit logs."}), 502


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Upload exceeds the 5 MB image size limit."}), 413


@app.get("/api/metrics")
def api_metrics():
    return jsonify(metrics_summary())


@app.get("/api/logs")
def api_logs():
    return jsonify({"calls": recent_calls()})


def _start_oauth(provider: str):
    if not app.secret_key:
        return jsonify({"error": "Set FLASK_SECRET_KEY before using OAuth routes."}), 503
    state = secrets.token_urlsafe(32)
    session[f"oauth_state_{provider}"] = state
    try:
        return redirect(authorization_url(provider, state))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


def _finish_oauth(provider: str):
    if not app.secret_key:
        return jsonify({"error": "Set FLASK_SECRET_KEY before using OAuth routes."}), 503
    expected_state = session.pop(f"oauth_state_{provider}", None)
    if not expected_state or not secrets.compare_digest(expected_state, request.args.get("state", "")):
        return jsonify({"error": "Invalid OAuth state."}), 400
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "OAuth provider returned no authorization code."}), 400
    try:
        exchange_code(provider, code)
    except Exception:
        app.logger.exception("OAuth exchange failed for %s", provider)
        return jsonify({"error": "OAuth exchange failed; check server logs and provider settings."}), 502
    return jsonify({"connected": provider})


@app.get("/auth/google/start")
def google_oauth_start():
    return _start_oauth("google")


@app.get("/auth/google/callback")
def google_oauth_callback():
    return _finish_oauth("google")


@app.get("/auth/notion/start")
def notion_oauth_start():
    return _start_oauth("notion")


@app.get("/auth/notion/callback")
def notion_oauth_callback():
    return _finish_oauth("notion")


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), debug=False)