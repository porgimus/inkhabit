# SPDX-License-Identifier: AGPL-3.0-or-later
"""A local page for trying the handwriting: type text, pick what it is for, move the sliders.

`inkhabit playground <out>` serves it on this machine only (127.0.0.1). It only reads the library;
nothing is changed and nothing leaves the computer.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import library, render

SLIDERS = [  # name, label, min, max, step, neutral
    ("size", "Size", 8, 30, 1, 16),
    ("letter_spacing", "Letter spacing", 0.6, 1.8, 0.05, 1.0),
    ("word_spacing", "Word spacing", 0.5, 2.0, 0.05, 1.0),
    ("line_spacing", "Line spacing", 0.7, 1.8, 0.05, 1.0),
    ("letter_height", "Letter height", 0.7, 1.5, 0.05, 1.0),
    ("letter_width", "Letter width", 0.7, 1.5, 0.05, 1.0),
    ("slant", "Slant (degrees)", -10, 20, 1, 0),
    ("weight", "Pen thickness", 0.4, 1.8, 0.05, 1.0),
    ("human", "Irregularity", 0, 2.5, 0.1, 1.0),
    ("joins", "Joined letters", 0, 1.5, 0.05, None),
    ("whole_words", "Whole words from notes", 0, 1, 0.05, None),
]

PAGE = """<!doctype html><meta charset="utf-8"><title>Try the handwriting</title>
<style>
body{font:15px/1.4 -apple-system,system-ui,sans-serif;margin:0;background:#f6f4ef;color:#222;display:grid;grid-template-columns:340px 1fr;height:100vh}
aside{background:#fffdf8;border-right:1px solid #ddd;padding:18px 20px;overflow:auto}
h1{font-size:17px;margin:0 0 12px}label{display:block;font-size:12.5px;color:#555;margin:11px 0 3px}
textarea{width:100%;height:150px;font:14px/1.4 ui-monospace,Menlo,monospace;padding:8px;border:1px solid #bbb;border-radius:8px;box-sizing:border-box}
select,button{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid #bbb;background:#fff}
button{cursor:pointer}button.primary{background:#1d4ed8;color:#fff;border-color:#1d4ed8}
input[type=range]{width:100%}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
.val{float:right;color:#222;font-variant-numeric:tabular-nums}
main{padding:28px;overflow:auto}#paper{background:#fff;border:1px solid #e0dccf;border-radius:10px;padding:26px;min-height:300px;box-shadow:0 2px 10px #0001}
#paper.dark{background:#2a2a2c}#cmd{font:12px ui-monospace,Menlo,monospace;background:#1c1917;color:#e7e5e4;padding:10px 12px;border-radius:8px;margin-top:16px;white-space:pre-wrap;word-break:break-all}
small{color:#777}
</style>
<aside><h1>Try the handwriting</h1>
<label>Text</label><textarea id="text">Dear Sam, thanks for the notes.
Bring the calculator on Friday at 9:30!</textarea>
<label>What is it for?</label><select id="for"></select>
<div class="row"><button class="primary" id="again">Another take</button><span id="seedv"><small>take 1</small></span>
<label style="margin:0"><input type="checkbox" id="justify"> justify</label>
<label style="margin:0"><input type="checkbox" id="dark"> dark paper</label></div>
<div id="sliders"></div>
<div class="row"><button id="reset">Reset sliders</button><a id="dl" download="handwriting.svg"><button>Download SVG</button></a></div>
</aside>
<main><div id="paper"></div><div id="cmd"></div>
<p><small>The grey box is the command that produces exactly what you see, so you or an agent can repeat it.</small></p></main>
<script>
const CFG=__CFG__;let seed=1,timer=null;const el=id=>document.getElementById(id);
for(const s of CFG.situations){const o=document.createElement("option");o.value=s[0];o.textContent=s[1];el("for").appendChild(o)}
const box=el("sliders");
for(const [name,label,min,max,step,neutral] of CFG.sliders){const w=document.createElement("div");
 w.innerHTML=`<label>${label}<span class="val" id="v_${name}"></span></label><input type="range" id="s_${name}" min="${min}" max="${max}" step="${step}">`;
 box.appendChild(w)}
function defaults(){for(const [name,,,,,neutral] of CFG.sliders){const d=CFG.defaults[el("for").value][name];el("s_"+name).value=(d!==undefined?d:neutral);el("s_"+name).dataset.d=el("s_"+name).value}}
function params(){const p=new URLSearchParams({text:el("text").value,for:el("for").value,seed,justify:el("justify").checked?1:0,dark:el("dark").checked?1:0});
 for(const [name] of CFG.sliders){const s=el("s_"+name);el("v_"+name).textContent=(+s.value).toFixed(s.step<1?2:0);if(s.value!==s.dataset.d||name=="size")p.set(name,s.value)}
 return p}
async function draw(){const p=params();const r=await fetch("/write?"+p);const j=await r.json();
 el("paper").innerHTML=j.svg;el("paper").className=el("dark").checked?"dark":"";el("cmd").textContent=j.command+(j.missing?"\\n# no shape for: "+j.missing:"");
 el("dl").href=URL.createObjectURL(new Blob([j.svg],{type:"image/svg+xml"}));el("seedv").innerHTML="<small>take "+seed+"</small>"}
function soon(){clearTimeout(timer);timer=setTimeout(draw,140)}
el("for").onchange=()=>{defaults();draw()};el("again").onclick=()=>{seed++;draw()};el("reset").onclick=()=>{defaults();draw()};
for(const id of ["text","justify","dark"])el(id).oninput=soon;box.oninput=soon;
defaults();draw();
</script>"""

SITUATION_LABELS = [("text", "Ordinary text (clean)"), ("official", "A form or official document (careful)"), ("heading", "A heading"), ("label", "A caption or margin note"),
                    ("letter", "A card or letter"), ("note", "Like my own notes"), ("scribble", "A quick scribble")]


def serve(out: str | Path, port: int = 8766) -> None:
    out = Path(out)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            lib = library.load(out / "library.json")
            if url.path == "/":
                defaults = {}
                for name, _ in SITUATION_LABELS:
                    h, size = render.situation(lib, name)
                    defaults[name] = {"size": round(16 * size), "letter_spacing": h.letter_spacing, "word_spacing": h.word_spacing,
                                      "line_spacing": h.line_spacing, "letter_height": h.letter_height, "letter_width": h.letter_width,
                                      "slant": h.slant, "weight": round(h.weight, 2), "joins": round(h.joins, 2),
                                      "whole_words": h.whole_words, "human": 1.0}
                cfg = json.dumps({"sliders": SLIDERS, "situations": SITUATION_LABELS, "defaults": defaults})
                self._send(PAGE.replace("__CFG__", cfg).encode(), "text/html; charset=utf-8")
            elif url.path == "/write":
                q = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
                name = q.get("for", "text")
                h, size = render.situation(lib, name if name in render.SITUATIONS else "text")
                flags = [f"--for {name}"]
                if "human" in q and float(q["human"]) != 1.0:
                    h = h.scaled(float(q["human"]))
                    flags.append(f"--human {q['human']}")
                sets = {k: float(q[k]) for k in ("letter_spacing", "word_spacing", "line_spacing", "letter_height", "letter_width",
                                                  "slant", "weight", "joins", "whole_words") if k in q}
                h = h.with_settings(sets)
                flags += [f"--set {k}={v:g}" for k, v in sets.items()]
                xh = float(q.get("size", 16 * size))
                seed = int(q.get("seed", 1))
                text = q.get("text", "")[:4000]
                ink = render.Writer(lib, seed, h).write(text, xh, 900, justify=q.get("justify") == "1")
                svg = render.to_svg(ink, "#f2f2f4" if q.get("dark") == "1" else "#1a1a2e")
                shown = text.replace("\n", "\\n").replace('"', '\\"')
                cmd = (f'inkhabit write {out}/library.json "{shown}" -o out.png --seed {seed} --x-height {xh / size:g} --width 900 '
                       + " ".join(flags) + (" --justify" if q.get("justify") == "1" else ""))
                self._send(json.dumps({"svg": svg, "command": cmd, "missing": "".join(sorted(ink.missing))}).encode(), "application/json")
            else:
                self.send_error(404)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"playground: http://127.0.0.1:{port}/   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
