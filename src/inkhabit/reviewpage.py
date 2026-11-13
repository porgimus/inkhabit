# SPDX-License-Identifier: AGPL-3.0-or-later
"""A local page for reviewing a library by eye: click a sample to drop it, press Apply.

`inkhabit review-page <out>` serves it on this machine only (127.0.0.1). Nothing leaves the computer.
Apply removes the marked samples, keeps a backup of the library, and rebuilds the font and sheets.
"""

from __future__ import annotations

import json
import time
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import library

PAGE = """<!doctype html><meta charset="utf-8"><title>Review handwriting</title>
<style>
:root{color-scheme:light}
body{font:15px/1.4 -apple-system,system-ui,sans-serif;margin:0;background:#f6f4ef;color:#222}
header{position:sticky;top:0;background:#fffdf8;border-bottom:1px solid #ddd;padding:12px 24px;display:flex;gap:16px;align-items:center;z-index:2}
header h1{font-size:17px;margin:0 12px 0 0}
button{font:inherit;padding:7px 14px;border-radius:8px;border:1px solid #bbb;background:#fff;cursor:pointer}
button.primary{background:#1d4ed8;color:#fff;border-color:#1d4ed8}button:disabled{opacity:.45;cursor:default}
nav button.on{background:#222;color:#fff;border-color:#222}
main{padding:16px 24px 80px;max-width:1280px;margin:auto}
p.help{color:#555;max-width:820px}
.row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #e6e2d8}
.lab{width:74px;font:600 26px ui-monospace,Menlo,monospace;color:#9a3412;text-align:center}
.lab small{display:block;font:11px -apple-system,sans-serif;color:#888}
.tile{width:104px;height:104px;background:#fff;border:2px solid #e3dfd3;border-radius:10px;cursor:pointer;position:relative;flex:none}
.tile:hover{border-color:#999}.tile.drop{border-color:#dc2626;background:#fee2e2}
.tile.drop::after{content:"drop";position:absolute;right:6px;top:4px;font:600 11px sans-serif;color:#b91c1c}
.tile svg{width:100%;height:100%}.cl{position:absolute;left:6px;bottom:3px;font:11px sans-serif;color:#888}
.note{font-size:12px;color:#777;margin-left:6px}
#msg{color:#166534;font-weight:600}
</style>
<header><h1>Review: __FAMILY__</h1>
<nav id="tabs"></nav><span style="flex:1"></span><span id="msg"></span>
<button id="clear">Clear marks</button><button id="apply" class="primary" disabled>Apply</button></header>
<main><p class="help">Each row is one character and its samples. Click any sample that is <b>not clearly that character</b>,
is cut off, has a stray mark, or just does not look like how you write it. Then press Apply: marked samples are removed and the
font is rebuilt. Faint lines show the baseline and the height of a small letter. The first three in each row are the ones
clean writing prefers. A small number in a corner is a clarity score from 0 to 100, where one has been measured.</p><div id="rows"></div></main>
<script>
const DATA=__DATA__;
const groups={"a-z":c=>/^[a-z]$/.test(c),"A-Z":c=>/^[A-Z]$/.test(c),"0-9":c=>/^[0-9]$/.test(c),
 "marks":c=>c.length==1&&!/[A-Za-z0-9]/.test(c),"joined pairs":c=>c.length==2};
let tab="a-z";const marks={};
function key(kind,ch,i){return kind+"\\u0001"+ch+"\\u0001"+i}
function svg(v){const s=24,ox=52-v.width*s/2,oy=64;let d="";
 for(const st of v.strokes){if(st.length==1){d+=`M${ox+st[0][0]*s-0.6},${oy+st[0][1]*s} l1.2,0 `}
  else d+="M"+st.map(p=>(ox+p[0]*s).toFixed(1)+","+(oy+p[1]*s).toFixed(1)).join(" L")+" "}
 return `<svg viewBox="0 0 104 104"><line x1="6" x2="98" y1="64" y2="64" stroke="#e5e0d2"/><line x1="6" x2="98" y1="40" y2="40" stroke="#f0ecdf"/>
 <path d="${d}" fill="none" stroke="#1c1917" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`}
function draw(){const el=document.getElementById("rows");el.innerHTML="";
 for(const r of DATA.rows.filter(r=>groups[tab](r.ch))){const row=document.createElement("div");row.className="row";
  row.innerHTML=`<div class="lab">${r.ch==" "?"space":r.ch.replace(/&/g,"&amp;").replace(/</g,"&lt;")}<small>${r.count?r.count+" found":r.source||""}</small></div>`;
  r.variants.forEach((v,i)=>{const t=document.createElement("div");const k=key(r.kind,r.ch,i);
   t.className="tile"+(marks[k]?" drop":"");t.innerHTML=svg(v)+(v.clarity!=null?`<span class="cl">${Math.round(v.clarity*100)}</span>`:"");t.title="sample "+i+(v.from?" from “"+v.from+"”":"");
   t.onclick=()=>{marks[k]?delete marks[k]:marks[k]=1;t.classList.toggle("drop");upd()};row.appendChild(t)});
  el.appendChild(row)}}
function upd(){const n=Object.keys(marks).length;const b=document.getElementById("apply");b.disabled=!n;b.textContent=n?`Apply (drop ${n})`:"Apply"}
const tabs=document.getElementById("tabs");
for(const g of Object.keys(groups)){const b=document.createElement("button");b.textContent=g;b.onclick=()=>{tab=g;[...tabs.children].forEach(x=>x.classList.toggle("on",x==b));draw()};
 if(g==tab)b.className="on";tabs.appendChild(b)}
document.getElementById("clear").onclick=()=>{for(const k in marks)delete marks[k];draw();upd()};
document.getElementById("apply").onclick=async()=>{const b=document.getElementById("apply");b.disabled=true;b.textContent="Rebuilding...";
 const drops=Object.keys(marks).map(k=>{const [kind,ch,i]=k.split("\\u0001");return {kind,ch,i:+i}});
 const r=await fetch("/apply",{method:"POST",headers:{"Content-Type":"application/json","X-Token":DATA.token},body:JSON.stringify({drops})});
 const j=await r.json();sessionStorage.setItem("msg",j.message);location.reload()};
document.getElementById("msg").textContent=sessionStorage.getItem("msg")||"";sessionStorage.removeItem("msg");
draw();
</script>"""


def _rows(lib: dict) -> list[dict]:
    rows = [{"kind": "glyph", "ch": ch, "count": e.get("count", 0), "source": (e["variants"][0].get("source") if e["variants"] else ""),
             "variants": [{"strokes": v["strokes"], "width": v["width"], "from": v.get("from", ""), "clarity": v.get("clarity")}
                          for v in e["variants"]]}
            for ch, e in lib["glyphs"].items() if e["variants"]]
    rows += [{"kind": "pair", "ch": p, "count": 0, "source": "joined", "variants": [{"strokes": v["strokes"], "width": v["width"]} for v in vs]}
             for p, vs in lib.get("pairs", {}).items()]
    return rows


def serve(out: str | Path, family: str = "My Handwriting", port: int = 8765, token: str = "") -> None:
    out = Path(out)
    token = token or hex(int(time.time() * 1e6))[2:]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/":
                self.send_error(404)
                return
            lib = library.load(out / "library.json")
            data = json.dumps({"rows": _rows(lib), "token": token}, ensure_ascii=False)
            body = PAGE.replace("__DATA__", data.replace("</", "<\\/")).replace("__FAMILY__", html.escape(family or lib.get("meta", {}).get("family") or "My Handwriting")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            # only this page may change the library: same-machine address, and the token the page was given
            if self.path != "/apply" or self.headers.get("X-Token") != token or self.client_address[0] != "127.0.0.1":
                self.send_error(403)
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            drops: dict[str, list[int]] = {}
            pair_drops: dict[str, list[int]] = {}
            for d in req.get("drops", []):
                (pair_drops if d["kind"] == "pair" else drops).setdefault(d["ch"], []).append(int(d["i"]))
            from . import pipeline

            lib = library.load(out / "library.json")
            library.save(lib, out / f"library.before-review-{time.strftime('%H%M%S')}.json")
            library.drop_variants(lib, drops, pair_drops)
            if lib.get("filled"):
                from . import extend

                lib = extend.fill(lib)
            pipeline.finish(lib, out, family, log=lambda *a: None)
            n = sum(map(len, drops.values())) + sum(map(len, pair_drops.values()))
            msg = f"Dropped {n} sample{'s' if n != 1 else ''}; font rebuilt."
            body = json.dumps({"message": msg}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"review page: http://127.0.0.1:{port}/   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
