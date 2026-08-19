"""Minimal web chat demo on the stdlib http.server -- no web framework dependency.

Single-user, single-process, intended for eyeballing a checkpoint, not deployment.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import ConversationEngine, GenerationConfig

PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>smalltalk-ai</title>
<style>
:root{color-scheme:light dark;--b:#8884}
body{font:15px/1.5 system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px}
h1{font-size:15px;font-weight:600;opacity:.6;margin:0 0 12px}
#log{display:flex;flex-direction:column;gap:8px;min-height:50vh;margin-bottom:12px}
.m{padding:8px 12px;border-radius:14px;max-width:78%;white-space:pre-wrap;word-break:break-word}
.u{align-self:flex-end;background:#3b82f6;color:#fff;border-bottom-right-radius:4px}
.a{align-self:flex-start;background:#8882;border-bottom-left-radius:4px}
form{display:flex;gap:8px;position:sticky;bottom:0;padding:8px 0;background:Canvas}
input[type=text]{flex:1;padding:10px 12px;border:1px solid var(--b);border-radius:20px;
 background:transparent;color:inherit;font:inherit}
button{padding:10px 16px;border:0;border-radius:20px;background:#3b82f6;color:#fff;font:inherit;cursor:pointer}
details{margin-top:16px;font-size:13px;opacity:.75}
label{display:flex;justify-content:space-between;gap:8px;margin:6px 0}
label input{width:90px}
</style>
<h1 id="hdr">smalltalk-ai</h1>
<div id="log"></div>
<form id="f"><input type="text" id="t" placeholder="say something..." autocomplete="off" autofocus>
<button>send</button></form>
<details><summary>decoding &amp; controls</summary>
<label>temperature <input type="number" id="temperature" step="0.05" value="0.7"></label>
<label>top_p <input type="number" id="top_p" step="0.05" value="0.9"></label>
<label>top_k <input type="number" id="top_k" step="1" value="0"></label>
<label>repetition_penalty <input type="number" id="repetition_penalty" step="0.05" value="1.1"></label>
<label>max_new_tokens <input type="number" id="max_new_tokens" step="4" value="48"></label>
<button type="button" id="reset">reset conversation</button>
</details>
<script>
const log=document.getElementById('log');
function add(text,cls){const d=document.createElement('div');d.className='m '+cls;
 d.textContent=text;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);}
function params(){const o={};for(const k of ['temperature','top_p','top_k',
 'repetition_penalty','max_new_tokens'])o[k]=parseFloat(document.getElementById(k).value);return o;}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
 const i=document.getElementById('t');const msg=i.value.trim();if(!msg)return;i.value='';
 add(msg,'u');
 const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},
  body:JSON.stringify({message:msg,gen:params()})});
 const j=await r.json();add(j.reply,'a');};
document.getElementById('reset').onclick=async()=>{await fetch('/api/reset',{method:'POST'});
 log.innerHTML='';};
fetch('/api/info').then(r=>r.json()).then(j=>{
 document.getElementById('hdr').textContent=`smalltalk-ai — ${j.params.toLocaleString()} params`;});
</script>
"""


def make_handler(engine: ConversationEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(
                    {
                        "params": engine.model.num_parameters(),
                        "vocab_size": engine.tokenizer.vocab_size,
                        "layers": engine.model.cfg.num_layers,
                        "state_enabled": engine.state is not None,
                    }
                )
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if self.path == "/api/reset":
                engine.reset()
                self._json({"ok": True})
                return
            if self.path != "/api/chat":
                self._json({"error": "not found"}, 404)
                return
            try:
                payload = json.loads(raw or b"{}")
                msg = str(payload.get("message", "")).strip()
                if not msg:
                    self._json({"error": "empty message"}, 400)
                    return
                overrides = {
                    k: v for k, v in (payload.get("gen") or {}).items()
                    if k in GenerationConfig().__dict__ and v is not None
                }
                if "top_k" in overrides:
                    overrides["top_k"] = int(overrides["top_k"])
                if "max_new_tokens" in overrides:
                    overrides["max_new_tokens"] = int(overrides["max_new_tokens"])
                reply = engine.reply(msg, gen=engine.gen.with_(**overrides))
                self._json({"reply": reply, "state": engine.state.to_dict() if engine.state else None})
            except Exception as exc:  # keep the demo alive
                self._json({"error": str(exc)}, 500)

        def log_message(self, fmt, *a):  # quieter console
            return

    return Handler


def serve(engine: ConversationEngine, host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"smalltalk-ai web demo on http://{host}:{port}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
