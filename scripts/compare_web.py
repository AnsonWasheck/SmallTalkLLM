#!/usr/bin/env python3
"""Side-by-side blind comparison of two or more checkpoints in the browser.

    python scripts/compare_web.py \
        --model A=/path/to/snap-r001 \
        --model B=/path/to/snap-r002 \
        --tokenizer artifacts/core/tokenizer-4096 --port 8010

One input box, every model answers the same turn, columns update together. Each
model keeps its OWN conversation history, so a multi-turn exchange is a fair test
of each model rather than one model reading another's replies.

Labels are deliberately opaque (A, B, ...) and the identity of each column is
hidden behind a reveal toggle: the point is to judge the replies before knowing
which checkpoint produced them. Core-Bench measures reflex accuracy and is blind
to conversational feel, which is exactly what this tool is for.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import _bootstrap  # noqa: F401

from smalltalk.infer.generate import GenerationConfig, load_engine

PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>smalltalk-ai :: side by side</title>
<style>
:root{color-scheme:light dark;--b:#8884}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1200px;margin:0 auto}
h1{font-size:14px;font-weight:600;opacity:.55;margin:0 0 4px}
.sub{font-size:13px;opacity:.5;margin:0 0 14px}
#cols{display:flex;gap:14px;align-items:stretch}
.col{flex:1;min-width:0;border:1px solid var(--b);border-radius:12px;padding:10px;display:flex;flex-direction:column}
.col h2{font:600 13px system-ui;margin:0 0 8px;opacity:.7;display:flex;justify-content:space-between}
.who{font-weight:400;opacity:.55;font-size:11px}
.log{display:flex;flex-direction:column;gap:7px;min-height:48vh;max-height:64vh;overflow-y:auto}
.m{padding:7px 11px;border-radius:13px;max-width:88%;white-space:pre-wrap;word-break:break-word;font-size:14px}
.u{align-self:flex-end;background:#3b82f6;color:#fff;border-bottom-right-radius:4px}
.a{align-self:flex-start;background:#8882;border-bottom-left-radius:4px}
.pending{opacity:.45;font-style:italic}
form{display:flex;gap:8px;margin-top:14px}
input[type=text]{flex:1;padding:11px 14px;border:1px solid var(--b);border-radius:22px;
 background:transparent;color:inherit;font:inherit}
button{padding:11px 18px;border:0;border-radius:22px;background:#3b82f6;color:#fff;font:inherit;cursor:pointer}
button.ghost{background:#8883;color:inherit}
.bar{display:flex;gap:8px;margin-top:10px;font-size:13px;align-items:center;flex-wrap:wrap}
details{margin-top:12px;font-size:13px;opacity:.8}
label{display:inline-flex;gap:6px;margin:4px 14px 4px 0;align-items:center}
label input{width:78px;padding:3px 6px;border:1px solid var(--b);border-radius:6px;background:transparent;color:inherit}
</style>
<h1>smalltalk-ai &mdash; side by side</h1>
<p class="sub">Same turn goes to every model. Each keeps its own history. Judge the replies before revealing which is which.</p>
<div id="cols"></div>
<form id="f"><input type="text" id="t" placeholder="say something to all of them..." autocomplete="off" autofocus>
<button>send</button></form>
<div class="bar">
  <button class="ghost" id="reveal" type="button">reveal identities</button>
  <button class="ghost" id="reset" type="button">reset all</button>
  <span id="note" style="opacity:.5"></span>
</div>
<details><summary>decoding</summary>
<label>temperature <input type="number" id="temperature" step="0.05" value="0"></label>
<label>top_p <input type="number" id="top_p" step="0.05" value="1.0"></label>
<label>repetition_penalty <input type="number" id="repetition_penalty" step="0.05" value="1.0"></label>
<label>max_new_tokens <input type="number" id="max_new_tokens" step="1" value="20"></label>
<p style="opacity:.6">temperature 0 = greedy, the setting Core-Bench scores under. Two runs give identical replies.</p>
</details>
<script>
let NAMES = [], revealed = false;
const cols = document.getElementById('cols');

function build(names){
  NAMES = names;
  cols.innerHTML = '';
  for(const n of names){
    const d = document.createElement('div');
    d.className = 'col';
    d.innerHTML = `<h2><span>Model ${n}</span><span class="who" id="who-${n}"></span></h2>
                   <div class="log" id="log-${n}"></div>`;
    cols.appendChild(d);
  }
}
function add(name, who, text, cls){
  const log = document.getElementById('log-'+name);
  const d = document.createElement('div');
  d.className = 'm ' + (who==='u'?'u':'a') + (cls?(' '+cls):'');
  d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}
function opts(){
  const g = id => parseFloat(document.getElementById(id).value);
  return {temperature:g('temperature'), top_p:g('top_p'),
          repetition_penalty:g('repetition_penalty'), max_new_tokens:g('max_new_tokens')};
}
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const t = document.getElementById('t');
  const msg = t.value.trim(); if(!msg) return;
  t.value = '';
  const holders = {};
  for(const n of NAMES){ add(n,'u',msg); holders[n] = add(n,'a','...','pending'); }
  const r = await fetch('/chat', {method:'POST', headers:{'content-type':'application/json'},
                                  body: JSON.stringify({text:msg, opts:opts()})});
  const data = await r.json();
  for(const n of NAMES){
    holders[n].textContent = data.replies[n];
    holders[n].classList.remove('pending');
  }
};
document.getElementById('reset').onclick = async () => {
  await fetch('/reset', {method:'POST'});
  for(const n of NAMES) document.getElementById('log-'+n).innerHTML='';
};
document.getElementById('reveal').onclick = async () => {
  revealed = !revealed;
  const r = await fetch('/identities'); const d = await r.json();
  for(const n of NAMES) document.getElementById('who-'+n).textContent = revealed ? d[n] : '';
  document.getElementById('reveal').textContent = revealed ? 'hide identities' : 'reveal identities';
};
fetch('/models').then(r=>r.json()).then(d=>{ build(d.names);
  document.getElementById('note').textContent = d.params + ' params each'; });
</script>
"""


def make_handler(engines: dict, identities: dict, params: str):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):        # keep the console readable
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/models":
                return self._send(200, json.dumps(
                    {"names": list(engines), "params": params}))
            if self.path == "/identities":
                return self._send(200, json.dumps(identities))
            self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/reset":
                with lock:
                    for e in engines.values():
                        e.reset()
                return self._send(200, json.dumps({"ok": True}))
            if self.path == "/chat":
                text = (payload.get("text") or "").strip()
                o = payload.get("opts") or {}
                replies = {}
                # Serialised: these share one GPU/CPU and one process. Sequential
                # generation also keeps the comparison honest -- no thread
                # scheduling differences leaking into what the user reads.
                with lock:
                    for name, eng in engines.items():
                        gen = eng.gen.with_(
                            temperature=float(o.get("temperature", 0.0)),
                            top_p=float(o.get("top_p", 1.0)),
                            repetition_penalty=float(o.get("repetition_penalty", 1.0)),
                            max_new_tokens=int(o.get("max_new_tokens", 20)),
                            greedy=float(o.get("temperature", 0.0)) <= 0.0,
                        )
                        replies[name] = eng.reply(text, gen=gen)
                return self._send(200, json.dumps({"replies": replies}))
            self._send(404, json.dumps({"error": "not found"}))

    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable, e.g. A=artifacts/runs/x/best")
    ap.add_argument("--tokenizer", default="artifacts/core/tokenizer-4096")
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps the GPU free for training (default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=20, seed=0)
    engines, identities, n_params = {}, {}, None
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects LABEL=PATH, got {spec!r}")
        print(f"[load] {label} <- {path}")
        eng = load_engine(path, args.tokenizer, device=args.device, gen=gen)
        engines[label] = eng
        identities[label] = path
        n_params = sum(p.numel() for p in eng.model.parameters())

    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(engines, identities, f"{n_params:,}"))
    print(f"\nside-by-side on http://{args.host}:{args.port}   (ctrl-c to stop)")
    print("identities are hidden until you press 'reveal'")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
