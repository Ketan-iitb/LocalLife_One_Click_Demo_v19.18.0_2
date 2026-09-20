"""Welcome page: the operator picks local, cloud or automatic before startup.

Styled to match the Accuracy Deployment v3.0 operator page so the two read as
one product. Everything technical -- ports, raw launcher output, cloud reasons
-- lives in the collapsed diagnostics block, not on the main card.
"""

WELCOME_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Welcome to Local Life Demo</title>
<style>
:root{--bg:#f4f7f6;--panel:#fff;--ink:#17221f;--muted:#61706b;--green:#188a64;--green2:#e7f6f0;--amber:#a46500;--amber2:#fff5df;--red:#b42318;--red2:#ffebe9;--blue:#1769aa;--line:#d9e2df;--shadow:0 8px 26px rgba(18,43,35,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,system-ui,sans-serif}
button{font:inherit;cursor:pointer}.app{max-width:1060px;margin:auto;padding:26px}
.brand h1{margin:0;font-size:30px}.brand p{margin:6px 0 0;color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:22px;flex-wrap:wrap}
.status-pill{padding:10px 16px;border-radius:999px;font-weight:800;border:1px solid var(--line);background:#fff}
.ok{color:var(--green);background:var(--green2);border-color:#b9e4d5}.warn{color:var(--amber);background:var(--amber2);border-color:#f1d89c}
.bad{color:var(--red);background:var(--red2);border-color:#f2beb9}.working{color:var(--blue);background:#eaf4fb;border-color:#bfdcf0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--shadow);display:flex;flex-direction:column}
.card h2{margin:0 0 6px;font-size:20px}.card .tag{font-size:12px;font-weight:800;color:var(--muted);letter-spacing:.04em}
.card ul{margin:12px 0 18px;padding-left:18px;color:var(--muted);line-height:1.65;font-size:14px}
.card button{margin-top:auto;padding:14px;border:0;border-radius:12px;font-weight:850;font-size:16px;background:var(--green);color:#fff}
.card button.secondary{background:#eef3f1;color:var(--ink);border:1px solid var(--line)}
.card button:disabled{opacity:.45;cursor:not-allowed}
.checks{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow);margin-bottom:18px}
.checkrow{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #edf1ef}.checkrow:last-child{border-bottom:0}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block;margin-right:9px;background:#99a5a1}
.dot.green{background:var(--green)}.dot.amber{background:#e0a21a}.dot.red{background:var(--red)}
.progress{margin-top:18px;background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow);display:none}
.progress.show{display:block}.log{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.7;color:var(--muted);white-space:pre-wrap}
.callout{border-radius:13px;padding:14px;margin:12px 0;background:var(--green2);border:1px solid #b9e4d5}
.callout.warn{background:var(--amber2);border-color:#efd590;color:#6e4700}.callout.bad{background:var(--red2);border-color:#f0bbb6;color:#7e2018}
details{margin-top:18px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
summary{cursor:pointer;font-weight:800}details pre{overflow:auto;font-size:12px;color:var(--muted)}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.text-btn{border:0;background:transparent;text-decoration:underline;color:var(--muted);padding:8px}
@media(max-width:860px){.cards{grid-template-columns:1fr}.app{padding:14px}}
</style></head>
<body><div class="app">
<div class="top">
  <div class="brand"><h1>Welcome to Local Life Demo</h1><p>How would you like to run the system?</p></div>
  <div id="overall" class="status-pill working">● Checking…</div>
</div>

<section class="checks">
  <div class="checkrow"><span><span id="dot-version" class="dot green"></span>Version</span><strong id="txt-version">—</strong></div>
  <div class="checkrow"><span><span id="dot-pi" class="dot"></span>Raspberry Pi</span><strong id="txt-pi">Checking</strong></div>
  <div class="checkrow"><span><span id="dot-rs" class="dot"></span>RealSense D435</span><strong id="txt-rs">Unknown until started</strong></div>
  <div class="checkrow"><span><span id="dot-log" class="dot"></span>Logitech C920</span><strong id="txt-log">Unknown until started</strong></div>
  <div class="checkrow"><span><span id="dot-net" class="dot"></span>Internet</span><strong id="txt-net">Checking</strong></div>
  <div class="checkrow"><span><span id="dot-cloud" class="dot"></span>Cloud configuration</span><strong id="txt-cloud">Checking</strong></div>
</section>

<section class="cards">
  <article class="card">
    <div class="tag">OPTION 1</div><h2>Run Locally</h2>
    <ul><li>Runs entirely on this laptop and the Raspberry Pi</li><li>Fastest and most reliable start</li><li>No cloud cost</li><li>Speed limited by local hardware</li><li>Same measurement pipeline as cloud mode</li></ul>
    <button onclick="choose('local')">RUN LOCALLY</button>
  </article>
  <article class="card">
    <div class="tag">OPTION 2</div><h2>Run with Cloud GPU</h2>
    <ul><li>Starts or connects to the Google Cloud GPU VM</li><li>Allows heavier segmentation and more frame aggregation</li><li>Needs internet, gcloud sign-in and GPU capacity</li><li>Takes longer to start</li><li><strong>Can create cloud charges</strong></li></ul>
    <button id="btn-cloud" onclick="choose('cloud')">RUN WITH CLOUD GPU</button>
  </article>
  <article class="card">
    <div class="tag">OPTION 3</div><h2>Automatic</h2>
    <ul><li>Checks whether the cloud is usable first</li><li>Uses the cloud GPU when it is healthy</li><li>Falls back to local when it is not</li><li>Records the mode actually used for every measurement</li></ul>
    <button class="secondary" onclick="choose('auto')">TRY CLOUD, FALL BACK TO LOCAL</button>
  </article>
</section>

<section id="progress" class="progress">
  <h2 id="progress-title">Starting…</h2>
  <div id="progress-note" class="callout">Please wait. This window shows each step as it completes.</div>
  <div id="progress-log" class="log"></div>
  <div class="row"><button class="text-btn" onclick="cancelStartup()">Cancel and go back</button><a class="text-btn" id="dash-link" style="display:none" href="/">Open the dashboard</a></div>
</section>

<details><summary>Diagnostics</summary><pre id="diag">—</pre></details>
</div>
<script>
const $=id=>document.getElementById(id); let ready=null, launch=null, polling=null;
async function api(url,payload){const r=await fetch(url,{method:payload===undefined?'GET':'POST',headers:payload===undefined?{}:{'Content-Type':'application/json'},body:payload===undefined?undefined:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
function setRow(id,value,good,bad,unknown){const dot=$('dot-'+id),txt=$('txt-'+id);if(value===null||value===undefined){dot.className='dot amber';txt.textContent=unknown||'Unknown';return}dot.className='dot '+(value?'green':'red');txt.textContent=value?good:bad}
function render(){if(!ready)return;
  $('txt-version').textContent=ready.version||'—';
  setRow('pi',ready.pi_reachable,'Reachable','Not reachable','Not configured');
  setRow('net',ready.internet,'Connected','Offline');
  setRow('cloud',ready.cloud_available,'Ready','Not available');
  $('btn-cloud').disabled=!ready.cloud_available;
  const good=ready.internet!==false;
  $('overall').className='status-pill '+(ready.cloud_available?'ok':(good?'warn':'bad'));
  $('overall').textContent=ready.cloud_available?'● READY':(good?'● LOCAL ONLY':'● OFFLINE');
  $('diag').textContent=JSON.stringify({readiness:ready,launch:launch},null,2);
  if(ready.launcher_script_error){$('overall').className='status-pill bad';$('overall').textContent='● SETUP PROBLEM';}
}
async function refresh(){try{const d=await api('/api/launcher/status');ready=d.readiness;launch=d.launch;render();renderLaunch()}catch(e){$('overall').className='status-pill bad';$('overall').textContent='● CONTROL SERVICE LOST'}}
function renderLaunch(){if(!launch||launch.phase==='idle')return;
  $('progress').classList.add('show');
  $('progress-title').textContent={checking:'Checking…','starting-cloud':'Starting cloud GPU…','starting-local':'Starting local pipeline…',running:'Running',failed:'Startup failed',cancelled:'Cancelled'}[launch.phase]||launch.phase;
  $('progress-log').textContent=(launch.messages||[]).join('\n');
  const note=$('progress-note');
  if(launch.phase==='failed'){note.className='callout bad';note.textContent=launch.error||'Startup failed.'}
  else if(launch.fallback_used){note.className='callout warn';note.textContent='Cloud was unavailable ('+(launch.cloud_error||'unknown reason')+'). Running locally instead.'}
  else if(launch.phase==='running'){note.className='callout';note.textContent='Processing mode: '+String(launch.mode||'').toUpperCase()+'. Opening the dashboard.'}
  else{note.className='callout';note.textContent='Please wait. This window shows each step as it completes.'}
  if(launch.phase==='running'&&launch.dashboard_url){const link=$('dash-link');link.href=launch.dashboard_url;link.style.display='inline';if(!polling){polling=true;setTimeout(()=>location.assign(launch.dashboard_url),1200)}}
}
async function choose(mode){
  let payload={mode};
  if(mode==='cloud'){if(!confirm('Starting the cloud GPU VM can create Google Cloud charges. The VM keeps billing until you run the stop tool.\n\nStart the cloud GPU now?'))return;payload.confirm_billing=true}
  try{const d=await api('/api/launcher/start',payload);launch=d.launch;renderLaunch();refresh()}
  catch(e){$('progress').classList.add('show');$('progress-note').className='callout bad';$('progress-note').textContent=e.message}
}
async function cancelStartup(){try{const d=await api('/api/launcher/cancel',{});launch=d.launch;renderLaunch()}catch(e){alert(e.message)}}
setInterval(refresh,1500);refresh();
</script></body></html>'''
