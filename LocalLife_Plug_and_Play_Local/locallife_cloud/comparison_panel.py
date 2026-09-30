"""Local-vs-Cloud panel shared by the operator page (/) and Research Mode (/research).

All numbers come from /api/local-cloud/view, which calculates them from the
durable store; the page only renders. Local and cloud are separate servers,
but the browser reaches both at the same origin (127.0.0.1:8000), so the page
keeps each mode's run summaries in localStorage and imports the other mode's
into the server it is talking to (attributed to their source host and run).
Wrapped in {% raw %} because both pages are Jinja templates.
"""

COMPARISON_PANEL = r"""{% raw %}
<section id="lc-panel" style="margin:14px 0;padding:16px;border:1px solid rgba(128,160,170,.35);border-radius:14px;background:rgba(17,29,35,.55)">
<style>
#lc-panel{font-size:13px}#lc-panel h2{margin:0 0 4px;font-size:18px}#lc-panel .lc-muted{opacity:.72}
#lc-panel table{width:100%;border-collapse:collapse;margin-top:8px}#lc-panel th,#lc-panel td{padding:5px 6px;border-bottom:1px solid rgba(128,160,170,.22);text-align:left;vertical-align:top}
#lc-panel th{font-size:12px;opacity:.8}#lc-panel .lc-cam td{font-weight:800;padding-top:12px;border-bottom:2px solid rgba(128,160,170,.4)}
#lc-panel .lc-good{color:#3fbf8a;font-weight:700}#lc-panel .lc-bad{color:#e0795c;font-weight:700}
#lc-panel .lc-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
#lc-panel select,#lc-panel input{background:transparent;color:inherit;border:1px solid rgba(128,160,170,.5);border-radius:7px;padding:5px}
#lc-panel select option{color:#111}#lc-panel a.lc-btn,#lc-panel button.lc-btn{border:1px solid rgba(128,160,170,.6);border-radius:8px;padding:7px 11px;text-decoration:none;color:inherit;background:rgba(80,120,130,.25);cursor:pointer;font-weight:700}
#lc-panel .lc-charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:10px}
#lc-panel .lc-chart{border:1px solid rgba(128,160,170,.3);border-radius:10px;padding:8px}#lc-panel svg{width:100%;height:150px}
@media(max-width:900px){#lc-panel .lc-charts{grid-template-columns:1fr}}
</style>
<h2>Local vs Cloud comparison</h2>
<div class="lc-muted" id="lc-status">Loading saved runs…</div>
<div class="lc-row">
 <label>Local run <select id="lc-local"></select></label>
 <label>Cloud run <select id="lc-cloud"></select></label>
 <label>Charts camera <select id="lc-cam"><option value="realsense">RealSense</option><option value="logitech">Logitech</option></select></label>
</div>
<div style="overflow-x:auto"><table id="lc-table"></table></div>
<div class="lc-charts">
 <div class="lc-chart"><strong>Server-side result latency p50 / p95 over runs (ms)</strong><div id="lc-ch-lat"></div></div>
 <div class="lc-chart"><strong>Completed frames / s over runs</strong><div id="lc-ch-fps"></div></div>
 <div class="lc-chart"><strong>Failure % and superseded % over runs</strong><div id="lc-ch-fail"></div></div>
</div>
<div class="lc-row">
 <a class="lc-btn" id="lc-dl-raw" href="/api/local-cloud/raw.csv">Download raw CSV</a>
 <a class="lc-btn" id="lc-dl-sum" href="/api/local-cloud/summaries.csv">Download run summary CSV</a>
 <a class="lc-btn" id="lc-dl-cmp" href="/api/local-cloud/comparison.csv">Download comparison CSV</a>
 <label>Filter downloads: run <select id="lc-filter-run"><option value="">all runs (history)</option></select></label>
 <label>since <input id="lc-filter-since" type="date"></label>
 <label class="lc-btn" style="font-weight:600">Import other mode's run summary CSV <input id="lc-import" type="file" accept=".csv" style="display:none"></label>
</div>
<div class="lc-muted" id="lc-notes"></div>
<script>
(function(){
const el=id=>document.getElementById(id);
const KEY='locallife_local_cloud_summaries_v1',SEL='locallife_local_cloud_selection_v1';
const fmt=v=>v==null?'N/A':(Math.abs(v)>=100?Number(v).toFixed(1):Number(v).toFixed(2));
const token=()=>{try{return typeof API_TOKEN==='undefined'?'':API_TOKEN}catch(e){return ''}};
const load=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))||d}catch(e){return d}};
const save=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}};
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let lastSync=0,view=null;
function runLabel(r){const d=r.started_wall?new Date(r.started_wall*1000).toLocaleString():'?';return r.processing_mode+' · '+d+' · '+(r.duration_s==null?'?':Math.round(r.duration_s)+' s')+' · '+r.source_host+(r.current?' (current)':'')+(r.source==='imported'?' (imported)':'')}
function fillSelect(sel,runs,mode,chosen){const opts=runs.filter(r=>r.processing_mode===mode);sel.innerHTML=opts.length?opts.map(r=>'<option value="'+esc(r.key)+'"'+(r.key===chosen?' selected':'')+'>'+esc(runLabel(r))+'</option>').join(''):'<option value="">No run yet</option>'}
async function sync(mode){
 // Cache this server's own runs; import cached runs of the other mode into it.
 try{const own=await (await fetch('/api/local-cloud/summaries')).json();const cache=load(KEY,{});
  for(const s of own.summaries||[]){const k=s.source_host+'|'+s.run_id+'|'+s.camera_id;if(!cache[k]||(cache[k].computed_at||0)<=(s.computed_at||0))cache[k]=s}
  save(KEY,cache);const other=Object.values(cache).filter(s=>s.processing_mode!==mode);
  if(other.length){const h={'Content-Type':'application/json'};if(token())h['X-API-Token']=token();await fetch('/api/local-cloud/import',{method:'POST',headers:h,body:JSON.stringify({summaries:other})})}
 }catch(e){}
}
function favour(row){const v=row.verdict||'—';const m=v.match(/^(cloud|local) better(.*)$/);return m?'<span class="'+(m[1]==='cloud'?'lc-good':'lc-bad')+'">'+m[1]+' better</span>'+esc(m[2]):esc(v)}
const na=(v,reason)=>v==null?'N/A'+(reason?'<div class="lc-muted" style="font-size:11px;max-width:220px">'+esc(reason)+'</div>':''):fmt(v);
function renderTable(v){let h='<tr><th>Metric and measured boundary</th><th>Unit</th><th>Local</th><th>Cloud</th><th>Cloud − Local</th><th>% diff (Cloud−Local)/Local</th><th>Favourable direction / verdict</th><th>n (L / C)</th><th>Run / window s (L / C)</th></tr>';let cam='';
 for(const r of v.rows){if(r.camera_id!==cam){cam=r.camera_id;h+='<tr class="lc-cam"><td colspan="9">'+(cam==='realsense'?'RealSense D435':'Logitech C920')+' · local machine '+esc(r.local_machine||'—')+' · cloud machine '+esc(r.cloud_machine||'—')+'<div class="lc-muted" style="font-weight:400">'+(r.matched?'Matched runs (same recorded input, models and length).':'NOT MATCHED — no better/worse verdict: '+esc(r.match_notes))+'</div></td></tr>'}
  h+='<tr><td>'+esc(r.label)+'<div class="lc-muted" style="font-size:11px;max-width:420px">'+esc(r.boundary)+'</div></td><td>'+esc(r.unit)+'</td><td>'+na(r.local,r.local_na_reason)+'</td><td>'+na(r.cloud,r.cloud_na_reason)+'</td><td>'+fmt(r.abs_difference)+'</td><td>'+(r.pct_difference==null?'N/A':fmt(r.pct_difference)+' %')+'</td><td>'+favour(r)+'</td><td>'+esc(r.local_n??'N/A')+' / '+esc(r.cloud_n??'N/A')+'</td><td>'+fmt(r.local_duration_s)+' / '+fmt(r.local_window_s)+'<br>'+fmt(r.cloud_duration_s)+' / '+fmt(r.cloud_window_s)+'</td></tr>'}
 h+='<tr><td>Detection / colour / volume accuracy</td><td>—</td><td colspan="7">'+esc(v.quality)+'</td></tr>';el('lc-table').innerHTML=h}
function chart(target,series,unit){
 // series: [{name, mode, points:[{x,y}]}]; x = run order; missing values leave gaps.
 const all=series.flatMap(s=>s.points.filter(p=>p.y!=null));const W=320,H=150,P=28;
 if(!all.length){el(target).innerHTML='<div class="lc-muted" style="padding:40px 0;text-align:center">No run yet</div>';return}
 const xs=all.map(p=>p.x),ys=all.map(p=>p.y);const x0=Math.min(...xs),x1=Math.max(...xs,x0+1),y1=Math.max(...ys)*1.1||1;
 const X=x=>P+(W-P-8)*(x-x0)/(x1-x0),Y=y=>H-P+4-(H-P-4)*y/y1;
 let s='<svg viewBox="0 0 '+W+' '+H+'"><line x1="'+P+'" y1="'+(H-P+4)+'" x2="'+W+'" y2="'+(H-P+4)+'" stroke="currentColor" opacity=".3"/><text x="2" y="12" font-size="9" fill="currentColor">'+fmt(y1)+' '+unit+'</text><text x="'+P+'" y="'+(H-6)+'" font-size="9" fill="currentColor">run number (oldest → newest; hover a point for run id and n)</text>';
 const colours={local:'#4aa3ff',cloud:'#f0a13a'};let legend='';
 for(const se of series){const pts=se.points.filter(p=>p.y!=null);const c=colours[se.mode];legend+='<span style="color:'+c+'">■ '+esc(se.name)+(pts.length?'':' — No run yet')+'</span> ';
  let path='';for(const p of se.points){if(p.y==null){path+=' ';continue}path+=(path.endsWith(' ')||!path?'M':'L')+X(p.x).toFixed(1)+','+Y(p.y).toFixed(1)}
  s+='<path d="'+path.trim()+'" fill="none" stroke="'+c+'" stroke-width="1.6"'+(se.dash?' stroke-dasharray="4 3"':'')+'/>';for(const p of pts)s+='<circle cx="'+X(p.x).toFixed(1)+'" cy="'+Y(p.y).toFixed(1)+'" r="2.6" fill="'+c+'"><title>'+esc(se.name)+': '+fmt(p.y)+' '+unit+' · run '+esc(p.run)+' · n='+esc(p.n)+'</title></circle>'}
 el(target).innerHTML=s+'</svg><div style="font-size:11px">'+legend+'</div>'}
let camPicked=false;function renderCharts(v){if(!camPicked){const has=c=>v.summaries.some(s=>s.camera_id===c);if(!has(el('lc-cam').value)&&has('logitech'))el('lc-cam').value='logitech';}
 const perMode=m=>v.summaries.filter(s=>s.processing_mode===m&&s.camera_id===el('lc-cam').value).length;
 if(Math.max(perMode('local'),perMode('cloud'))<2){for(const t of ['lc-ch-lat','lc-ch-fps','lc-ch-fail'])el(t).innerHTML='<div class="lc-muted" style="padding:30px 8px;text-align:center">Trend needs at least two independently identified runs of one mode. With one run per mode, use the table above.</div>';return}
 if(!camPicked){const has=c=>v.summaries.some(s=>s.camera_id===c);if(!has(el('lc-cam').value)&&has('logitech'))el('lc-cam').value='logitech';}const cam=el('lc-cam').value;const sums=v.summaries.filter(s=>s.camera_id===cam);const order=v.runs.map(r=>r.key);
 const pts=(mode,key)=>sums.filter(s=>s.processing_mode===mode).map(s=>({x:order.indexOf(s.source_host+'|'+s.run_id),y:s[key],run:s.run_id,n:s.latency_n}));
 chart('lc-ch-lat',[{name:'Local p50',mode:'local',points:pts('local','latency_p50_ms')},{name:'Local p95',mode:'local',dash:1,points:pts('local','latency_p95_ms')},{name:'Cloud p50',mode:'cloud',points:pts('cloud','latency_p50_ms')},{name:'Cloud p95',mode:'cloud',dash:1,points:pts('cloud','latency_p95_ms')}],'ms');
 chart('lc-ch-fps',[{name:'Local',mode:'local',points:pts('local','fps')},{name:'Cloud',mode:'cloud',points:pts('cloud','fps')}],'frames/s');
 chart('lc-ch-fail',[{name:'Local failure %',mode:'local',points:pts('local','failure_pct')},{name:'Local superseded %',mode:'local',dash:1,points:pts('local','superseded_pct')},{name:'Cloud failure %',mode:'cloud',points:pts('cloud','failure_pct')},{name:'Cloud superseded %',mode:'cloud',dash:1,points:pts('cloud','superseded_pct')}],'%')}
function downloads(v){const run=el('lc-filter-run').value,since=el('lc-filter-since').value;const q=[];if(run)q.push('run_id='+encodeURIComponent(run));if(since)q.push('since='+(Date.parse(since)/1000));
 el('lc-dl-raw').href='/api/local-cloud/raw.csv'+(q.length?'?'+q.join('&'):'');el('lc-dl-sum').href='/api/local-cloud/summaries.csv'+(run?'?run_id='+encodeURIComponent(run):'');
 el('lc-dl-cmp').href='/api/local-cloud/comparison.csv?local_run='+encodeURIComponent(v.selected.local||'')+'&cloud_run='+encodeURIComponent(v.selected.cloud||'')}
async function refresh(){try{
 if(view&&Date.now()-lastSync>60000){lastSync=Date.now();await sync(view.processing_mode)}
 const s=load(SEL,{});const q='?local_run='+encodeURIComponent(s.local||'')+'&cloud_run='+encodeURIComponent(s.cloud||'');
 const v=await (await fetch('/api/local-cloud/view'+q)).json();view=v;
 if(!lastSync){lastSync=Date.now();await sync(v.processing_mode)}
 fillSelect(el('lc-local'),v.runs,'local',v.selected.local);fillSelect(el('lc-cloud'),v.runs,'cloud',v.selected.cloud);
 const fr=el('lc-filter-run'),cur=fr.value;fr.innerHTML='<option value="">all runs (history)</option>'+[...new Set(v.runs.filter(r=>r.source==='own').map(r=>r.run_id))].map(id=>'<option'+(id===cur?' selected':'')+'>'+esc(id)+'</option>').join('');
 el('lc-status').innerHTML='This server processes in <b>'+esc(v.processing_mode.toUpperCase())+'</b> mode · '+esc(v.other_mode_status)+' · '+v.runs.length+' saved run(s) · '+v.counts.raw_observations+' raw observations';
 renderTable(v);renderCharts(v);downloads(v);
 el('lc-notes').innerHTML='Boundaries: '+Object.entries(v.boundaries).map(([k,b])=>'<b>'+esc(k)+'</b> '+esc(b)).join(' · ')+'. % diff = 100 × (Cloud − Local) / Local; N/A when Local is 0 or a value is missing. Superseded frames were replaced by a newer frame by design and are not errors. Unavailable metrics are N/A, never 0.';
}catch(e){el('lc-status').textContent='Comparison unavailable: '+e.message}}
el('lc-local').onchange=el('lc-cloud').onchange=()=>{save(SEL,{local:el('lc-local').value,cloud:el('lc-cloud').value});refresh()};
el('lc-cam').onchange=()=>{camPicked=true;view&&renderCharts(view)};el('lc-filter-run').onchange=el('lc-filter-since').onchange=()=>view&&downloads(view);
el('lc-import').onchange=async ev=>{const f=ev.target.files[0];if(!f)return;const h={'Content-Type':'text/csv'};if(token())h['X-API-Token']=token();const r=await fetch('/api/local-cloud/import',{method:'POST',headers:h,body:await f.text()});const d=await r.json();alert(r.ok?('Imported '+d.accepted+' summaries ('+d.rejected+' rejected)'):(d.error||'Import failed'));refresh()};
// Not window 'load': the live MJPEG streams keep the page loading forever.
const start=()=>{refresh();setInterval(refresh,5000)};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start);else start();
})();
</script>
</section>
{% endraw %}"""
