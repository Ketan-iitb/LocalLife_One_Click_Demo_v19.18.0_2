"""'Local vs Cloud: Cost and Accuracy' -- an independent dashboard section (renders /api/cost-accuracy/view).

Separate from the existing Local-vs-Cloud performance panel, which is unchanged. Loads on open and when
a selection or rate changes (no refresh loop, no inference). Wrapped in {% raw %} for the Jinja pages.
"""

COST_ACCURACY_PANEL = r"""{% raw %}
<section id="ca-panel" style="margin:14px 0;padding:16px;border:1px solid rgba(128,160,170,.35);border-radius:14px;background:rgba(17,29,35,.55)">
<style>
#ca-panel{font-size:13px}#ca-panel h2{margin:0 0 6px;font-size:18px}#ca-panel .ca-muted{opacity:.75}
#ca-panel table{width:100%;border-collapse:collapse;margin-top:8px}#ca-panel th,#ca-panel td{padding:4px 6px;border-bottom:1px solid rgba(128,160,170,.22);text-align:left;vertical-align:top}
#ca-panel th{font-size:12px;opacity:.8}#ca-panel .ca-scroll{overflow:auto}
#ca-panel select,#ca-panel input{background:transparent;color:inherit;border:1px solid rgba(128,160,170,.5);border-radius:7px;padding:4px}
#ca-panel select option{color:#111}#ca-panel button,#ca-panel a.ca-btn{border:1px solid rgba(128,160,170,.6);border-radius:8px;padding:6px 10px;color:inherit;background:rgba(80,120,130,.25);cursor:pointer;font-weight:700;text-decoration:none}
#ca-panel .ca-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:10px}
#ca-panel .ca-card{border:1px solid rgba(128,160,170,.3);border-radius:10px;padding:10px;min-width:0}
#ca-panel .ca-form label{display:inline-block;margin:3px 10px 3px 0}#ca-panel .ca-form input{width:110px}
#ca-panel svg text{fill:currentColor}
@media(max-width:900px){#ca-panel .ca-grid{grid-template-columns:1fr}}
</style>
<h2>Local vs Cloud: Cost and Accuracy</h2>
<div class="ca-muted">Cost uses measured runtime and processed frames with the rates entered below; accuracy needs paired local/cloud replays of the same recorded frames and an independent reference CSV. Nothing is estimated where data is missing.</div>
<div style="margin:8px 0">
 <label>Camera <select id="ca-camera"><option value="realsense">RealSense D435</option><option value="logitech">Logitech C920</option></select></label>
 <label>Local run <select id="ca-local"></select></label>
 <label>Cloud run <select id="ca-cloud"></select></label>
 <button type="button" id="ca-refresh">Refresh</button>
 <a class="ca-btn" id="ca-csv" href="/api/cost-accuracy.csv">Download cost &amp; accuracy CSV</a>
</div>
<div id="ca-status" class="ca-muted">Loading…</div>
<div class="ca-scroll"><table><thead><tr><th>Mode</th><th>Run</th><th>Matched / eligible trials</th><th>Valid volume</th><th>Volume MAE (L)</th><th>Volume MAPE (%)</th><th>Detection precision / recall</th><th>Processed FPS</th><th>Latency p50 / p95 (ms)</th><th>Evaluated runtime</th><th>Run cost</th><th>Cost / 1,000 frames</th></tr></thead><tbody id="ca-rows"></tbody></table></div>
<div class="ca-grid">
 <div class="ca-card"><b>Volume MAE vs cost per 1,000 processed frames</b><div id="ca-chart-acc"></div></div>
 <div class="ca-card"><b>Throughput vs cost per 1,000 processed frames</b> <span class="ca-muted">(speed, not accuracy)</span><div id="ca-chart-fps"></div></div>
</div>
<div class="ca-grid">
 <div class="ca-card"><b>Cost breakdown</b><div id="ca-breakdown" class="ca-muted"></div></div>
 <div class="ca-card"><b>Assumptions &amp; sources</b><ul id="ca-assumptions" class="ca-muted" style="margin:4px 0;padding-left:18px"></ul><div id="ca-resources" class="ca-muted"></div></div>
</div>
<div class="ca-card" style="margin-top:10px"><b>Cloud billing (actual, from the project's Google Cloud bill)</b> <span id="ca-bill-period" class="ca-muted"></span>
<div class="ca-grid" style="margin-top:4px"><div><div class="ca-muted">Cost per running hour by setup (SEK/h)</div><div id="ca-chart-rates"></div></div><div><div class="ca-muted">Where the money went</div><div id="ca-chart-spend"></div></div></div>
<ul id="ca-bill-notes" class="ca-muted" style="margin:6px 0;padding-left:18px"></ul></div>
<details style="margin-top:10px"><summary><b>Rates and local inputs</b> (edit; affects cost only)</summary>
<div style="margin:6px 0"><label>Billed setup <select id="ca-preset"></select></label> <button type="button" id="ca-preset-apply">Use this setup's billed rates</button></div>
<form id="ca-form" class="ca-form" style="margin-top:6px"></form>
<div id="ca-form-msg" class="ca-muted"></div>
</details>
<script>
(function(){
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const na=(v,d,u)=>v==null?'<span class="ca-muted">N/A</span>':(typeof v==='number'?v.toFixed(d):esc(v))+(u||'');
const headers=extra=>{const h=Object.assign({},extra||{});try{if(API_TOKEN)h['X-API-Token']=API_TOKEN;}catch(e){}return h;};
let view=null;
function query(){const p=new URLSearchParams({camera:$('ca-camera').value});if($('ca-local').value)p.set('local',$('ca-local').value);if($('ca-cloud').value)p.set('cloud',$('ca-cloud').value);return p.toString();}
function fill(sel,opts,chosen){sel.innerHTML=(opts||[]).map(o=>'<option value="'+esc(o.key)+'"'+(o.key===chosen?' selected':'')+'>'+esc(o.label)+'</option>').join('')||'<option value="">no run</option>';}
function scatter(points,yKey,yLabel,cur,empty){
 const pts=(points||[]).filter(p=>p.x_cost_per_1000!=null&&p[yKey]!=null);
 if(!pts.length)return '<div class="ca-muted" style="padding:18px 4px">'+esc(empty)+'</div>';
 const W=440,H=260,L=60,R=20,T=16,B=46,xm=Math.max(...pts.map(p=>p.x_cost_per_1000))*1.25||1,ym=Math.max(...pts.map(p=>p[yKey]))*1.25||1;
 const X=v=>L+(W-L-R)*v/xm,Y=v=>H-B-(H-T-B)*v/ym;let g='';
 for(let i=0;i<=4;i++){const yv=ym*i/4,xv=xm*i/4;g+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+Y(yv)+'" y2="'+Y(yv)+'" stroke="rgba(128,160,170,.25)"/><text x="'+(L-6)+'" y="'+(Y(yv)+4)+'" text-anchor="end" font-size="11">'+yv.toFixed(2)+'</text><text x="'+X(xv)+'" y="'+(H-B+16)+'" text-anchor="middle" font-size="11">'+xv.toFixed(4)+'</text>';}
 for(const p of pts){const c=p.mode==='local'?'#4ea1ff':'#ff7b72';
  const tip=['camera '+p.camera,'mode '+p.mode,'samples '+(p.n??'N/A'),'volume availability '+(p.availability_pct==null?'N/A':p.availability_pct.toFixed(1)+'%'),'cost basis '+p.cost_basis,'cost/1000 '+p.x_cost_per_1000.toFixed(4)+' '+cur,'MAE '+(p.y_mae_l==null?'N/A':p.y_mae_l.toFixed(3)+' L'),'FPS '+(p.y_fps==null?'N/A':p.y_fps.toFixed(2)),'latency p50 '+(p.latency_p50_ms==null?'N/A':p.latency_p50_ms.toFixed(0)+' ms')].join('\n');
  g+='<g><title>'+esc(tip)+'</title><circle cx="'+X(p.x_cost_per_1000)+'" cy="'+Y(p[yKey])+'" r="7" fill="'+c+'"/><text x="'+(X(p.x_cost_per_1000)+10)+'" y="'+(Y(p[yKey])-9)+'" font-size="12">'+esc(p.mode)+'</text></g>';}
 return '<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="max-width:'+W+'px" role="img" aria-label="'+esc(yLabel)+'">'+g+'<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="currentColor"/><line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="currentColor"/><text x="'+((L+W-R)/2)+'" y="'+(H-10)+'" text-anchor="middle" font-size="12">cost per 1,000 processed frames ('+esc(cur)+')</text><text x="14" y="'+((T+H-B)/2)+'" text-anchor="middle" font-size="12" transform="rotate(-90 14 '+((T+H-B)/2)+')">'+esc(yLabel)+'</text></svg><div class="ca-muted">'+pts.length+' evaluated configuration(s); measured points only, no trend line.</div>';}
function bars(rows,unit,stack){/* rows: [{label, parts:[{name,value,color}]}] -- horizontal bars, values written on them */
 const W=600,rowH=26,L=300,R=80,H=rows.length*rowH+24,max=Math.max(...rows.map(r=>r.parts.reduce((a,p)=>a+p.value,0)))||1;let g='';
 rows.forEach((r,i)=>{let x=L;const y=8+i*rowH;g+='<text x="'+(L-6)+'" y="'+(y+15)+'" text-anchor="end" font-size="11">'+esc(r.label)+'</text>';
  for(const p of r.parts){const w=(W-L-R)*p.value/max;g+='<rect x="'+x+'" y="'+y+'" width="'+Math.max(w,0.5)+'" height="'+(rowH-8)+'" fill="'+p.color+'"><title>'+esc(r.label+' — '+p.name+': '+p.value+' '+unit)+'</title></rect>';x+=w;}
  const tot=r.parts.reduce((a,p)=>a+p.value,0);g+='<text x="'+(x+5)+'" y="'+(y+14)+'" font-size="11">'+(Math.round(tot*100)/100)+' '+esc(unit)+'</text>';});
 const legend=stack?stack.map((s,i)=>'<span style="display:inline-block;width:10px;height:10px;background:'+s[1]+';margin:0 4px 0 '+(i?12:0)+'px"></span>'+esc(s[0])).join(''):'';
 return '<svg viewBox="0 0 '+W+' '+H+'" width="100%" style="max-width:'+W+'px">'+g+'</svg>'+(legend?'<div class="ca-muted" style="font-size:11px">'+legend+'</div>':'');}
function billing(b){if(!b)return;$('ca-bill-period').textContent='('+b.period+': total '+b.total+' '+b.currency+'; August '+b.by_month.August+', September '+b.by_month.September+'; budget '+b.budget_per_month+' '+b.currency+'/month)';
 $('ca-chart-rates').innerHTML=bars(Object.entries(b.presets).map(([k,p])=>({label:k.replace(' + 1x ',' + '),parts:[{name:'GPU',value:p.gpu_rate_per_hour,color:'#ff7b72'},{name:'CPU + RAM',value:p.machine_rate_per_hour,color:'#4ea1ff'}]})),'SEK/h',[['GPU','#ff7b72'],['CPU + RAM','#4ea1ff']]);
 $('ca-chart-spend').innerHTML=bars(b.breakdown.map(([k,v])=>({label:k+' ('+Math.round(100*v/b.total)+'%)',parts:[{name:k,value:v,color:k.startsWith('Disk')||k.startsWith('Images')||k.startsWith('Bucket')?'#d29922':'#8b949e'}]})),'SEK',[['storage','#d29922'],['compute / network','#8b949e']]);
 $('ca-bill-notes').innerHTML=b.notes.map(n=>'<li>'+esc(n)+'</li>').join('')+'<li>GPU time by type: '+b.gpu_breakdown.map(([k,v])=>esc(k)+' '+v+' SEK').join(', ')+'.</li><li>Source: '+esc(b.source)+'</li>';
 const sel=$('ca-preset');if(!sel.options.length)sel.innerHTML=Object.keys(b.presets).map(k=>'<option>'+esc(k)+'</option>').join('');}
function money(v,cur){return v==null?'<span class="ca-muted">unknown</span>':(v!==0&&Math.abs(v)<0.01?v.toPrecision(3):v.toFixed(4))+' '+esc(cur);}
function render(v){view=v;const cur=v.currency;
 fill($('ca-local'),v.options.local,v.selected.local);fill($('ca-cloud'),v.options.cloud,v.selected.cloud);
 $('ca-csv').href='/api/cost-accuracy.csv?'+query();
 $('ca-status').innerHTML=(v.accuracy_status==='evaluated'?'Accuracy evaluated on '+v.paired_frames+' paired frames. ':'<b>'+esc(v.accuracy_status)+'</b>. ')+esc(v.volume_definition);
 $('ca-rows').innerHTML=v.rows.map(r=>!r.available?'<tr><td>'+esc(r.mode)+'</td><td colspan="11" class="ca-muted">'+esc(r.reason)+'</td></tr>':
  '<tr><td><b>'+esc(r.mode)+'</b></td><td>'+esc(r.run_kind)+'<br><span class="ca-muted">'+esc(r.run_id)+'</span></td><td>'+(r.eligible==null?'<span class="ca-muted">not evaluated</span>':r.paired_frames+' / '+r.eligible)+'</td><td>'+(r.eligible==null?'<span class="ca-muted">N/A</span>':r.valid_volume+' of '+r.eligible+' ('+na(r.availability_pct,1,'%')+'), missing '+r.missing_volume)+'<br><span class="ca-muted">failed frames '+na(r.failed_frames,0)+'</span></td><td>'+na(r.mae_l,3)+'</td><td>'+na(r.mape_pct,1)+(r.mape_n!=null?'<br><span class="ca-muted">n='+r.mape_n+(r.zero_reference_excluded?', '+r.zero_reference_excluded+' zero-reference excluded':'')+'</span>':'')+'</td><td>'+(r.precision==null&&r.recall==null?'<span class="ca-muted">'+esc(r.detection_note||'not evaluated')+'</span>':na(r.precision,3)+' / '+na(r.recall,3))+'</td><td>'+na(r.fps,2)+'<br><span class="ca-muted">'+na(r.processed_frames,0)+' processed</span></td><td>'+na(r.latency_p50_ms,0)+' / '+na(r.latency_p95_ms,0)+'<br><span class="ca-muted">'+esc(r.latency_boundary)+'</span></td><td>'+(r.runtime_s==null?'<span class="ca-muted">N/A</span>':(r.runtime_s/60).toFixed(2)+' min')+'<br><span class="ca-muted">'+esc(r.runtime_basis)+'</span></td><td>'+money(r.run_cost,cur)+'<br><span class="ca-muted">'+esc(r.cost_kind)+'</span></td><td>'+money(r.cost_per_1000,cur)+'</td></tr>').join('');
 $('ca-chart-acc').innerHTML=scatter(v.points,'y_mae_l','Volume MAE (L) — lower is better',cur,v.accuracy_status==='evaluated'?'No point: a run has no known cost (enter rates below).':v.accuracy_status);
 $('ca-chart-fps').innerHTML=scatter(v.points,'y_fps','Processed FPS — higher is faster',cur,'No point: a run has no known cost (enter rates / local inputs below).');
 const b=v.breakdown,l=b.local,c=b.cloud;let h='';
 if(c)h+='<div><b>Cloud</b>: compute '+money(c.compute,cur)+' ('+esc(c.compute_rate_note)+(c.compute_rate_per_hour!=null?', '+c.compute_rate_per_hour.toFixed(2)+' '+esc(cur)+'/h':'')+'; '+(c.hours==null?'hours N/A':c.hours.toFixed(4)+' h, '+esc(c.hours_basis))+') · storage '+money(c.storage,cur)+' ('+esc(c.storage_basis)+') · other '+money(c.other,cur)+' · <b>total '+money(c.total,cur)+'</b> ('+esc(c.total_kind)+'). Excluded: '+esc(c.excluded)+'</div>';
 if(l)h+='<div style="margin-top:4px"><b>Local</b>: energy '+money(l.energy,cur)+' ('+esc(l.energy_basis)+(l.hours==null?'':'; '+l.hours.toFixed(4)+' h')+') · hardware amortisation '+money(l.hardware_allocated,cur)+' ('+esc(l.hardware_basis)+') · <b>total '+money(l.total,cur)+'</b> ('+esc(l.total_kind)+')</div>';
 $('ca-breakdown').innerHTML=h||'No runs selected.';
 $('ca-assumptions').innerHTML=v.assumptions.map(a=>'<li>'+esc(a)+'</li>').join('');
 const r=v.resources||{};$('ca-resources').innerHTML=r.available?'Configured in gpu.py (not the live VM): L4 machine types '+esc((r.l4_machine_types||[]).join(', '))+'; T4 fallback '+esc(r.t4_machine_type)+'; boot disk '+esc(r.disk_gb)+' GB '+esc(r.disk_type)+'; '+esc(r.provisioning)+'; '+esc(r.zones)+'; '+esc(r.stopped_vm_note)+'.':'gpu.py not found: resources unknown.';
 form(v.config);billing(v.billing);}
function form(cfg){const f=$('ca-form');if(f.dataset.ready)return;f.dataset.ready='1';
 const cl=cfg.cloud,lo=cfg.local;const inp=(sec,k,label,val,type)=>'<label>'+esc(label)+' <input data-sec="'+sec+'" data-key="'+k+'" type="'+(type||'text')+'" value="'+esc(val??'')+'"></label>';
 f.innerHTML='<div><b>Cloud</b> ('+esc(cl.status)+')</div>'+inp('cloud','source_url','Official source URL',cl.source_url)+inp('cloud','retrieved_on','Retrieved (YYYY-MM-DD)',cl.retrieved_on)+inp('cloud','region','Region',cl.region)+inp('cloud','provisioning','Provisioning',cl.provisioning)+inp('cloud','machine_type','Machine type',cl.machine_type)+inp('cloud','machine_rate_per_hour','Machine rate /h',cl.machine_rate_per_hour,'number')+inp('cloud','gpu_type','GPU type',cl.gpu_type)+inp('cloud','gpu_count','GPU count',cl.gpu_count,'number')+inp('cloud','gpu_rate_per_hour','GPU rate /h',cl.gpu_rate_per_hour,'number')+'<label>GPU included in machine rate <select data-sec="cloud" data-key="gpu_included_in_machine_rate"><option value=""'+(cl.gpu_included_in_machine_rate==null?' selected':'')+'>not stated</option><option value="true"'+(cl.gpu_included_in_machine_rate===true?' selected':'')+'>yes</option><option value="false"'+(cl.gpu_included_in_machine_rate===false?' selected':'')+'>no (billed separately)</option></select></label>'+inp('cloud','disk_type','Disk type',cl.disk_type)+inp('cloud','disk_gb','Disk GB',cl.disk_gb,'number')+inp('cloud','disk_rate_per_gb_month','Disk rate /GB-month',cl.disk_rate_per_gb_month,'number')+inp('cloud','billable_hours_override','Billable VM hours (optional)',cl.billable_hours_override,'number')+inp('cloud','other_per_hour','Other /h',cl.other_per_hour,'number')+inp('cloud','billed_cost','Actual billed cost (optional)',cl.billed_cost,'number')+
 '<div style="margin-top:6px"><b>Local</b></div>'+inp('local','power_w','Average power W',lo.power_w,'number')+'<label>Power measured <select data-sec="local" data-key="power_measured"><option value="false"'+(!lo.power_measured?' selected':'')+'>no (assumed)</option><option value="true"'+(lo.power_measured?' selected':'')+'>yes</option></select></label>'+inp('local','tariff_per_kwh','Tariff per kWh',lo.tariff_per_kwh,'number')+inp('local','hardware_cost','Hardware cost',lo.hardware_cost,'number')+inp('local','hardware_lifetime_hours','Lifetime operating h',lo.hardware_lifetime_hours,'number')+
 '<div style="margin-top:6px"></div>'+inp('top','currency','Currency',cfg.currency)+inp('top','exchange_rate_note','Exchange rate source/date',cfg.exchange_rate_note)+inp('top','truth_csv','Reference CSV path (optional)',cfg.truth_csv)+' <button type="submit">Save rates</button>';}
$('ca-form').addEventListener('submit',async e=>{e.preventDefault();const body={cloud:{},local:{}};
 for(const el of $('ca-form').querySelectorAll('[data-key]')){const v=el.value;if(el.dataset.sec==='top')body[el.dataset.key]=v;else body[el.dataset.sec][el.dataset.key]=v;}
 const res=await fetch('/api/cost-accuracy/config',{method:'POST',headers:headers({'Content-Type':'application/json'}),body:JSON.stringify(body)});
 const r=await res.json().catch(()=>({}));$('ca-form-msg').textContent=res.ok?'Saved ('+r.config.cloud.status+').':'Not saved: '+(r.error||res.status);if(res.ok){$('ca-form').dataset.ready='';load();}});
$('ca-preset-apply').addEventListener('click',async()=>{const res=await fetch('/api/cost-accuracy/preset',{method:'POST',headers:headers({'Content-Type':'application/json'}),body:JSON.stringify({name:$('ca-preset').value})});const r=await res.json().catch(()=>({}));$('ca-form-msg').textContent=res.ok?'Rates set from the bill: '+$('ca-preset').value:'Not applied: '+(r.error||res.status);if(res.ok){$('ca-form').dataset.ready='';load();}});
async function load(){try{const res=await fetch('/api/cost-accuracy/view?'+query(),{cache:'no-store'});const v=await res.json();if(!res.ok)throw new Error(v.error||res.status);render(v);}catch(err){$('ca-status').textContent='Cost & accuracy data unavailable: '+err.message;}}
$('ca-camera').addEventListener('change',()=>{$('ca-local').innerHTML='';$('ca-cloud').innerHTML='';load();});
$('ca-local').addEventListener('change',load);$('ca-cloud').addEventListener('change',load);$('ca-refresh').addEventListener('click',load);
load();
})();
</script>
</section>
{% endraw %}"""
