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
  <div id="pi-hint" class="callout warn" style="display:none;white-space:pre-line"></div>
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

  <div id="stage-block" style="display:none">
    <div class="checkrow"><span><strong id="stage-name">—</strong></span><strong id="stage-timing">—</strong></div>
    <div id="stage-list" class="log"></div>
  </div>

  <div id="cloud-facts" class="checks" style="display:none;margin-top:14px;box-shadow:none">
    <div class="checkrow"><span>VM</span><strong id="f-vm">—</strong></div>
    <div class="checkrow"><span>Zone</span><strong id="f-zone">—</strong></div>
    <div class="checkrow"><span>External IP</span><strong id="f-ip">—</strong></div>
    <div class="checkrow"><span>SSH host key</span><strong id="f-ssh">—</strong></div>
    <div class="checkrow"><span>Deployment version</span><strong id="f-deploy">—</strong></div>
    <div class="checkrow"><span>Cloud backend</span><strong id="f-backend">—</strong></div>
    <div class="checkrow"><span>Tunnel</span><strong id="f-tunnel">—</strong></div>
    <div class="checkrow"><span>Raspberry Pi</span><strong id="f-pi">—</strong></div>
    <div class="checkrow"><span>RealSense stream</span><strong id="f-rs">—</strong></div>
    <div class="checkrow"><span>Logitech stream</span><strong id="f-lg">—</strong></div>
    <div class="checkrow"><span>First frame</span><strong id="f-frame">—</strong></div>
  </div>

  <div id="progress-log" class="log"></div>
  <div class="row">
    <button class="text-btn" onclick="cancelStartup()">Cancel and go back</button>
    <button class="text-btn" id="btn-retry" style="display:none" onclick="choose(lastMode||'cloud')">Retry cloud</button>
    <button class="text-btn" id="btn-local" style="display:none" onclick="choose('local')">Run locally instead</button>
    <button class="text-btn" id="btn-diag" style="display:none" onclick="document.querySelector('details').open=true">Open diagnostics</button>
    <a class="text-btn" id="dash-link" style="display:none" href="/">Open the dashboard</a>
  </div>
</section>

@@METRICS_SECTION@@

<details><summary>Diagnostics</summary><pre id="diag">—</pre></details>
</div>
<script>
const $=id=>document.getElementById(id); let ready=null, launch=null, polling=null, startup=null, checklist=null, lastMode=null;
// Readable labels for the startup state machine. The page never parses console
// text -- every one of these comes from the structured status endpoint.
const STAGE_LABEL={idle:'Idle',resolving_zone:'Resolving zone',starting_vm:'Starting GPU VM',verifying_ssh:'Verifying SSH host key',checking_deployment:'Checking deployment',starting_backend:'Starting cloud backend',opening_tunnel:'Opening secure tunnel',connecting_pi:'Connecting Raspberry Pi',waiting_for_frames:'Waiting for first frame',ready:'Ready'};
const FAILURE_LABEL={gpu_unavailable:'No GPU capacity available',vm_start_timeout:'The VM did not start in time',ssh_host_key_mismatch:'SSH host key did not match the key Google published for this VM',ssh_authentication_failed:'SSH sign-in failed',deployment_failed:'Deploying the project to the VM failed',backend_health_failed:'The cloud backend never reported healthy',tunnel_failed:'The secure tunnel could not be opened',pi_unreachable:'The Raspberry Pi could not be reached',camera_stream_timeout:'No camera frames arrived',cloud_disconnected:'The cloud connection dropped'};
// Three states, never two: a check that has not run yet says so rather than
// showing a red cross the operator would read as a fault.
function mark(id,value,good,bad){const el=$(id);if(el===null)return;if(value===null||value===undefined){el.textContent='Unknown';el.style.color='';return}el.textContent=value?good:bad;el.style.color=value?'var(--green)':'var(--red)'}
async function api(url,payload){const r=await fetch(url,{method:payload===undefined?'GET':'POST',headers:payload===undefined?{}:{'Content-Type':'application/json'},body:payload===undefined?undefined:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
function setRow(id,value,good,bad,unknown){const dot=$('dot-'+id),txt=$('txt-'+id);if(value===null||value===undefined){dot.className='dot amber';txt.textContent=unknown||'Unknown';return}dot.className='dot '+(value?'green':'red');txt.textContent=value?good:bad}
function render(){if(!ready)return;
  $('txt-version').textContent=ready.version||'—';
  setRow('pi',ready.pi_reachable,'Reachable'+(ready.pi_address?' at '+ready.pi_address.host:''),'Not reachable','Not configured');
  if(ready.pi_hint){$('txt-pi').title=ready.pi_hint;$('pi-hint').textContent=ready.pi_hint;$('pi-hint').style.display='block'}else{$('pi-hint').style.display='none'}
  setRow('net',ready.internet,'Connected','Offline');
  setRow('cloud',ready.cloud_available,'Ready','Not available');
  $('btn-cloud').disabled=!ready.cloud_available;
  const good=ready.internet!==false;
  $('overall').className='status-pill '+(ready.cloud_available?'ok':(good?'warn':'bad'));
  $('overall').textContent=ready.cloud_available?'● READY':(good?'● LOCAL ONLY':'● OFFLINE');
  $('diag').textContent=JSON.stringify({readiness:ready,launch:launch},null,2);
  if(ready.launcher_script_error){$('overall').className='status-pill bad';$('overall').textContent='● SETUP PROBLEM';}
}
async function refresh(){try{const d=await api('/api/launcher/status');ready=d.readiness;launch=d.launch;startup=d.startup;checklist=d.checklist;render();renderLaunch();renderStartup()}catch(e){$('overall').className='status-pill bad';$('overall').textContent='● CONTROL SERVICE LOST'}}
function renderStartup(){if(!startup||(startup.stage==='idle'&&!startup.failure)){$('stage-block').style.display='none';$('cloud-facts').style.display='none';return}
  $('progress').classList.add('show');$('stage-block').style.display='block';$('cloud-facts').style.display='block';
  $('stage-name').textContent=(STAGE_LABEL[startup.stage]||startup.stage)+(startup.stage_index!==null?' ('+(startup.stage_index+1)+' of '+startup.stage_count+')':'');
  $('stage-timing').textContent=startup.stage_elapsed_seconds.toFixed(0)+'s in this step · '+startup.elapsed_seconds.toFixed(0)+'s total';
  $('stage-list').textContent=(startup.history||[]).map(h=>(STAGE_LABEL[h.stage]||h.stage)+': '+h.seconds+'s'+(h.complete?'':' (running)')).join('\n');
  const f=startup.facts||{};
  $('f-vm').textContent=f.vm_name||'—';
  $('f-zone').textContent=f.zone||f.preferred_zone||'—';
  // Shown because it is useful for diagnosis, labelled so nobody mistakes it
  // for proof of identity -- the verified host key is what proves that.
  $('f-ip').textContent=(f.external_ip||'—')+' (address, not identity)';
  $('f-ssh').textContent=f.ssh_verified===true?('Verified '+(f.ssh_fingerprint||'')):(f.ssh_verified===false?'NOT VERIFIED':'Unknown');
  $('f-ssh').style.color=f.ssh_verified===true?'var(--green)':(f.ssh_verified===false?'var(--red)':'');
  $('f-deploy').textContent=f.deployment_version?(f.deployment_matches===false?'Out of date ('+f.deployment_version+')':'Matches ('+f.deployment_version+')'):'Unknown';
  mark('f-backend',f.backend_healthy,'Healthy','Not healthy');
  mark('f-tunnel',f.tunnel_active,'Connected','Not connected');
  mark('f-pi',f.pi_connected,'Connected','Not connected');
  mark('f-rs',f.realsense_streaming,'Streaming','Not streaming');
  mark('f-lg',f.logitech_streaming,'Streaming','Not streaming');
  mark('f-frame',f.camera_heartbeat,'Received','Not received');
  const failed=!!startup.failure;
  for(const pair of [['btn-retry',failed],['btn-local',failed],['btn-diag',failed]]){$(pair[0]).style.display=pair[1]?'inline':'none'}
  if(failed){const note=$('progress-note');note.className='callout bad';
    note.textContent=(FAILURE_LABEL[startup.failure]||startup.failure)+' — '+(startup.reason||'')+' (after '+startup.elapsed_seconds.toFixed(0)+'s, at step: '+(STAGE_LABEL[startup.stage]||startup.stage)+')'}}
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
  lastMode=mode; let payload={mode};
  if(mode==='cloud'){if(!confirm('Starting the cloud GPU VM can create Google Cloud charges. The VM keeps billing until you run the stop tool.\n\nStart the cloud GPU now?'))return;payload.confirm_billing=true}
  try{const d=await api('/api/launcher/start',payload);launch=d.launch;renderLaunch();refresh()}
  catch(e){$('progress').classList.add('show');$('progress-note').className='callout bad';$('progress-note').textContent=e.message}
}
async function cancelStartup(){try{const d=await api('/api/launcher/cancel',{});launch=d.launch;renderLaunch()}catch(e){alert(e.message)}}
@@METRICS_JS@@
setInterval(refresh,1500);refresh();
</script></body></html>'''


# Local-versus-cloud comparison: shared by the welcome page and /compare, so
# the table is visible on its own page even when the demo was started
# directly with START_LOCAL_LIFE_CLOUD.cmd. Every number comes from the
# backend's own /api/telemetry; the mode of a column is the mode the backend
# reported, never the one that was requested.
METRICS_SECTION = r"""<section class="checks" id="metrics-panel">
  <h2 style="margin:0 0 4px">Local vs cloud comparison (automatic)</h2>
  <div class="muted" id="m-backend" style="font-size:13px">Waiting for a running pipeline…</div>
  <div style="overflow-x:auto"><table id="m-table" class="m-table"></table></div>
  <div class="muted" id="m-notes" style="font-size:12px;margin-top:6px"></div>
  <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap">
    <button class="text-btn" onclick="vmStatus()">Check VM (gpu.py status, no charge)</button>
    <a class="text-btn" id="m-csv" href="#" target="_blank">Export live telemetry CSV</a>
    <a class="text-btn" id="m-json" href="#" target="_blank">Export live telemetry JSON</a>
    <a class="text-btn" href="/compare" target="_blank">Open comparison on its own page</a>
  </div>
  <pre id="m-vm" style="display:none;font-size:12px;white-space:pre-wrap"></pre>
</section>"""

METRICS_CSS = """.m-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
.m-table th{text-align:left;padding:6px 8px;border-bottom:2px solid #cfd8d4}
.m-table td{padding:6px 8px;border-top:1px solid #edf1ef;vertical-align:top}
.m-table td:first-child{font-weight:650;white-space:nowrap}.m-table tr.sub td:first-child{font-weight:400;padding-left:22px}
.m-live{display:inline-block;font-size:11px;font-weight:800;padding:1px 7px;border-radius:9px;background:#e7f6f0;color:#188a64}
.m-old{display:inline-block;font-size:11px;font-weight:800;padding:1px 7px;border-radius:9px;background:#f1f3f2;color:#61706b}"""

METRICS_JS = r"""const NA='N/A';
const n=(v,d=1)=>v==null||isNaN(v)?NA:Number(v).toFixed(d);
function cellFor(snap,fn){if(!snap)return 'no run yet';const s=snap.summary;const parts=[fn(s.all,s,'all')].filter(Boolean);for(const [c,m] of Object.entries(s.cameras||{})){const v=fn(m,s,c);if(v)parts.push(c+': '+v)}return parts.join('<br>')}
const pct=(a,b)=>b?(100*a/b):null;
// [label, per-scope formatter, optional {get: all-scope number, better: 'lower'|'higher'} for the ratio column, sub-row]
const ROWS=[
 ['1. Latency p50 / p95 (ms)',m=>m.status!=='ok'?'insufficient data (n='+m.latency.n+')':n(m.latency.p50_ms)+' / '+n(m.latency.p95_ms)+' (n='+m.latency.n+', '+m.window_s+' s)',{get:s=>s.all.latency.p50_ms,better:'lower'}],
 ['Inference time p50 / p95 (ms)',m=>n(m.processing.p50_ms)+' / '+n(m.processing.p95_ms)+' (n='+m.processing.n+')',{get:s=>s.all.processing.p50_ms,better:'lower'},true],
 ['Edge upload RTT p50 / p95 (ms)',m=>n(m.client_upload_rtt.p50_ms)+' / '+n(m.client_upload_rtt.p95_ms)+' (n='+m.client_upload_rtt.n+')',{get:s=>s.all.client_upload_rtt.p50_ms,better:'lower'},true],
 ['2. Throughput (unique frames/s)',m=>m.throughput.fps==null?'insufficient data':n(m.throughput.fps,2)+' ('+m.throughput.unique_completed+' frames)',{get:s=>s.all.throughput.fps,better:'higher'}],
 ['3. Reliability completed / sent',m=>{const r=m.reliability;return (r.completed_pct==null?'insufficient data':n(r.completed_pct)+'%')+' ('+r.completed+'/'+r.sent+'; failed '+r.failed+', lost '+r.lost_in_transit+', reconnects '+r.reconnects+')'},{get:s=>s.all.reliability.completed_pct,better:'higher'}],
 ['Frames dropped for a newer frame',m=>{const r=m.reliability;const v=pct(r.superseded_by_newer_frame,r.received);return v==null?NA:n(v)+'% ('+r.superseded_by_newer_frame+'/'+r.received+')'},{get:s=>pct(s.all.reliability.superseded_by_newer_frame,s.all.reliability.received),better:'lower'},true],
 ['Frames received / in flight / duplicates',m=>{const r=m.reliability;return r.received+' / '+r.in_flight+' / '+r.duplicates_ignored},null,true],
 ['4. Measurement quality',()=> 'Not evaluated live (no ground truth); see replay benchmark'],
 ['5. Host CPU / RAM %',(m,s,c)=>c!=='all'?'':n(s.host_resources.cpu_percent)+' / '+n(s.host_resources.ram_percent)+' ('+(s.host.hostname||'?')+')'],
 ['GPU util % / VRAM MB',(m,s,c)=>c!=='all'?'':(s.gpu&&s.gpu.available?n(s.gpu.utilization_percent)+' / '+n(s.gpu.memory_used_mb,0)+' of '+n(s.gpu.memory_total_mb,0):NA),null,true],
 ['Processing hardware',(m,s,c)=>c!=='all'?'':((s.gpu&&s.gpu.available)?s.gpu.name:'CPU only (no GPU reported)')+' · '+((s.host&&s.host.cpu_count)||'?')+' CPU threads',null,true],
 ['Models in use',(m,s,c)=>c!=='all'?'':(s.models?('detector '+s.models.detector+' · Logitech depth '+s.models.logitech_depth+(s.models.logitech_depth_model?' ('+s.models.logitech_depth_model.split('/').pop()+')':'')):NA),null,true],
 ['Pi CPU / RAM %',(m,s,c)=>{const e=(s.edge_resources||{})[c];return c==='all'?'':(e?n(e.cpu_percent)+' / '+n(e.ram_percent):NA)},null,true],
 ['6. Bytes in / out per frame',m=>n(m.transfer.bytes_in_per_frame,0)+' / '+n(m.transfer.bytes_out_per_completed_frame,0),{get:s=>s.all.transfer.bytes_in_per_frame,better:'lower'}],
 ['Bytes in / out per minute',m=>n(m.transfer.bytes_in_per_minute,0)+' / '+n(m.transfer.bytes_out_per_minute,0),null,true],
];
function ratioCell(row,d){const spec=row[2];if(!spec||!d.modes.local||!d.modes.cloud)return '—';
 const l=spec.get(d.modes.local.summary),c=spec.get(d.modes.cloud.summary);
 if(l==null||c==null||!isFinite(l)||!isFinite(c)||l<=0||c<=0)return '—';
 if(spec.better==='lower'){return c<l?'cloud '+n(l/c,2)+'× lower':(c>l?'local '+n(c/l,2)+'× lower':'equal')}
 return c>l?'cloud '+n(c/l,2)+'× higher':(c<l?'local '+n(l/c,2)+'× higher':'equal')}
function headerFor(label,snap){if(!snap)return label;return label+' '+(snap.live?'<span class="m-live">LIVE</span>':'<span class="m-old">last seen '+Math.round(snap.age_s)+' s ago</span>')}
function benchCell(b){if(!b)return 'no replay run';const q=b.quality||{};const l=b.latency_e2e||{};return 'e2e p50/p95 '+n(l.p50_ms)+'/'+n(l.p95_ms)+' ms (n='+(l.n||0)+'), '+n(b.throughput_fps,2)+' fps; quality: '+(q.status||'Not evaluated')}
async function refreshMetrics(){try{const d=await api('/api/launcher/metrics');const b=d.backend;
 $('m-backend').innerHTML=(b.reachable?'Backend on port '+b.port+' reports processing mode <b>'+b.processing_mode.toUpperCase()+'</b>':'Backend not reachable ('+(b.error||'not started')+')')+(d.fallback_used?' · <b style="color:#b45309">cloud failed; running LOCAL fallback</b>':'')+(d.mode_mismatch?' · <b style="color:#b91c1c">MODE MISMATCH: launched '+d.launched_mode+'</b>':'');
 let h='<tr><th>Metric</th><th>'+headerFor('Local',d.modes.local)+'</th><th>'+headerFor('Cloud',d.modes.cloud)+'</th><th>Cloud vs local</th></tr>';
 for(const row of ROWS)h+='<tr'+(row[3]?' class="sub"':'')+'><td>'+row[0]+'</td><td>'+cellFor(d.modes.local,row[1])+'</td><td>'+cellFor(d.modes.cloud,row[1])+'</td><td>'+ratioCell(row,d)+'</td></tr>';
 h+='<tr><td>Replay benchmark (fair comparison)</td><td>'+benchCell(d.benchmarks.local)+'</td><td>'+benchCell(d.benchmarks.cloud)+'</td><td>—</td></tr>';
 h+='<tr><td>Cloud cost</td><td>—</td><td>'+(d.cost.estimate==null?NA:'≈ '+d.cost.estimate+' '+d.cost.currency+' (estimate)')+'</td><td>—</td></tr>';
 $('m-table').innerHTML=h;
 const any=(d.modes.local||d.modes.cloud);$('m-notes').textContent=(any?('Latency boundary: '+any.summary.all.latency.boundary+'. Inference boundary: '+any.summary.all.processing.boundary+'. Edge RTT boundary: '+any.summary.all.client_upload_rtt.boundary+'. Reliability denominator: '+any.summary.all.reliability.denominator+'. Live runs differ in scene and timing; the replay benchmark is the fair comparison. '):'')+d.cost.note;
 const base='http://127.0.0.1:'+b.port;$('m-csv').href=base+'/api/telemetry.csv';$('m-json').href=base+'/api/telemetry.json';
}catch(e){$('m-backend').textContent='Metrics unavailable: '+e.message}}
async function vmStatus(){const p=$('m-vm');p.style.display='block';p.textContent='Running gpu.py status…';try{const d=await api('/api/launcher/vm-status',{});p.textContent=(d.zone?'VM '+d.vm_state+' in zone '+d.zone+'\n\n':'')+(d.output||d.error||'')}catch(e){p.textContent=e.message}}
setInterval(refreshMetrics,3000);refreshMetrics();"""

WELCOME_PAGE = (WELCOME_PAGE.replace("@@METRICS_SECTION@@", METRICS_SECTION)
                .replace("@@METRICS_JS@@", METRICS_JS)
                .replace("</style>", METRICS_CSS + "\n</style>", 1))

COMPARE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local vs Cloud</title>
<style>
:root{--bg:#f4f7f6;--panel:#fff;--ink:#17221f;--muted:#61706b;--line:#d9e2df}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,system-ui,sans-serif}
.app{max-width:1200px;margin:auto;padding:16px}.muted{color:var(--muted)}
.checks{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:18px}
.text-btn{background:none;border:0;color:#1769aa;text-decoration:underline;cursor:pointer;font:inherit;padding:0}
""" + METRICS_CSS + """
</style></head><body><main class="app">
<h1 style="margin:4px 0 12px;font-size:26px">Local vs cloud comparison</h1>
""" + METRICS_SECTION + """
</main><script>
const $=id=>document.getElementById(id);
async function api(url,payload){const r=await fetch(url,{method:payload===undefined?'GET':'POST',headers:payload===undefined?{}:{'Content-Type':'application/json'},body:payload===undefined?undefined:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
""" + METRICS_JS + """
</script></body></html>"""
