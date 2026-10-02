"""v45 "Bin fill & new deposits": one independent card per camera (renders /api/bin-fill only).

State lives in the backend: a page refresh never resets the session or the counts.
Wrapped in {% raw %} because both pages are Jinja templates.
"""

BIN_FILL_PANEL = r"""{% raw %}
<section id="bf-panel" style="margin:14px 0;padding:16px;border:1px solid rgba(128,160,170,.35);border-radius:14px;background:rgba(17,29,35,.55)">
<style>
#bf-panel{font-size:13px}#bf-panel h2{margin:0 0 4px;font-size:18px}#bf-panel .bf-muted{opacity:.72}
#bf-panel .bf-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:8px}
#bf-panel .bf-card{border:1px solid rgba(128,160,170,.3);border-radius:10px;padding:10px;min-width:0}
#bf-panel .bf-kpis{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end}#bf-panel .bf-big{font-size:28px;font-weight:800;line-height:1.1}
#bf-panel .bf-na{color:#e0795c;font-weight:700}#bf-panel .bf-ok{color:#3fbf8a;font-weight:700}#bf-panel .bf-warn{color:#e0b85c}
#bf-panel table{width:100%;border-collapse:collapse;margin-top:8px}#bf-panel th,#bf-panel td{padding:4px 5px;border-bottom:1px solid rgba(128,160,170,.22);text-align:left;vertical-align:top}
#bf-panel th{font-size:12px;opacity:.8}#bf-panel .bf-scroll{max-height:260px;overflow:auto}
#bf-panel input,#bf-panel select{background:transparent;color:inherit;border:1px solid rgba(128,160,170,.5);border-radius:7px;padding:4px;width:80px}
#bf-panel select option{color:#111}#bf-panel button,#bf-panel a.bf-btn{border:1px solid rgba(128,160,170,.6);border-radius:8px;padding:6px 10px;color:inherit;background:rgba(80,120,130,.25);cursor:pointer;font-weight:700;text-decoration:none}
#bf-panel .bf-set label{display:inline-block;margin:3px 8px 3px 0}
@media(max-width:900px){#bf-panel .bf-grid{grid-template-columns:1fr}}
</style>
<h2>Bin fill &amp; new deposits — each camera independently</h2>
<div style="margin:6px 0"><button type="button" id="bf-reset">Reset counts &amp; readings</button> <a class="bf-btn" href="/api/session-deposits.xlsx">Download Excel</a> <a class="bf-btn" href="/api/session-deposits.csv">Download CSV</a></div>
<div id="bf-session" class="bf-muted">Loading…</div>
<div class="bf-grid"><div class="bf-card" id="bf-cam-realsense"></div><div class="bf-card" id="bf-cam-logitech"></div></div>
<div class="bf-muted" style="margin-top:6px">Fill = average waste height ÷ 100 cm bin height; litres = 660 L × fill (approximate).</div>
<div style="margin-top:8px"></div>

<script>
(function(){
const $=id=>document.getElementById(id);const bfHeaders=extra=>{const h=Object.assign({},extra||{});try{if(API_TOKEN)h['X-API-Token']=API_TOKEN;}catch(e){}return h;};const t=s=>s?new Date(s*1000).toLocaleTimeString():'—';const v=x=>x==null?'N/A':x;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const age=s=>s?Math.max(0,Math.round(Date.now()/1000-s))+' s ago':'—';
const name=id=>id==='realsense'?'RealSense D435':'Logitech C920';
const STATE={initialising:'Initialising baseline…',watching:'Watching for deposits',candidate:'Candidate: something entering',settling:'Settling…'};
function card(id,r,w){w=w||{};const cam=id==='realsense'?'RealSense':'Logitech';let h='<b>'+name(id)+'</b><div class="bf-kpis">';
 const ok=r.status==='ok';
 h+='<div><div class="bf-muted">'+cam+' bin fill (%)</div><div class="bf-big">'+(ok?r.height_fill_pct+'%':'—')+'</div>'+(ok&&r.stale?'<div class="bf-warn">stale · '+age(r.last_valid_at)+'</div>':'')+'</div>';
 h+='<div><div class="bf-muted">New bags this session</div><div class="bf-big">'+(w.new_bags||0)+'</div></div></div>';
 h+='<div class="'+(w.state==='watching'?'bf-ok':'bf-warn')+'">'+esc(STATE[w.state]||(w.state?w.state:'no frames yet'))+'</div>';
 if(ok)h+='<div class="bf-muted">Bin occupancy (approx., height-based): average surface '+r.max_fill_height_cm+' cm of '+r.usable_height_cm+' cm · ≈ '+r.rough_litres+' L filled / '+r.rough_remaining_litres+' L remaining of 660 L · tallest '+r.tallest_cm+' cm</div>';
 else{const g=r.diagnostics||{};h+='<div class="bf-muted">Bin fill unavailable: '+esc(r.reason)+(g.floor_source?' · floor: '+esc(g.floor_source)+(g.floor_distance_cm!=null?' '+g.floor_distance_cm+' cm':'')+' · valid depth '+v(g.valid_depth_pct)+'% · below floor '+v(g.below_floor_pct)+'% · above rim '+v(g.above_rim_pct)+'% · scale drift '+v(g.scale_drift):'')+'</div>';}
 const e=(w.events||[])[(w.events||[]).length-1];
 if(e)h+='<div style="margin-top:6px"><b>Latest new bag</b> ('+esc(e.event_id)+', '+t(e.deposit_time)+'): '+esc(e.detector_label&&e.detector_label!=='unknown'?e.detector_label:e.object_type)+' · '+(e.length_cm==null?'L×W N/A':e.length_cm+'×'+e.width_cm)+'×'+v(e.height_cm)+' cm · <b>'+cam+' object volume (L): '+v(e.envelope_l)+'</b> · colour '+esc(e.colour)+' · material '+esc(e.material)+' · '+esc(e.measurement_status)+(e.reason?' – '+esc(e.reason):'')+'</div>';
 h+='<div class="bf-muted">Last frame '+t(r.last_processed_at)+' · last valid fill '+t(r.last_valid_at)+' · last deposit '+t(w.last_confirmed_at)+'</div>';
 if(w.latest_rejection)h+='<div class="bf-muted">'+w.rejected+' rejected — latest: '+esc(w.latest_rejection)+'</div>';
 h+='<div class="bf-scroll"><table><thead><tr><th>Event</th><th>Time</th><th>Object</th><th>Colour</th><th>Material</th><th>L×W×H cm</th><th>'+cam+' object volume (L)</th><th>Status</th></tr></thead><tbody>';
 const ev=(w.events||[]).slice().reverse();
 for(const x of ev)h+='<tr><td>'+esc(x.event_id)+'</td><td>'+t(x.deposit_time)+'</td><td>'+esc(x.detector_label&&x.detector_label!=='unknown'?x.detector_label:x.object_type)+'</td><td>'+esc(x.colour)+'</td><td>'+esc(x.material)+'</td><td>'+(x.length_cm==null?'N/A':x.length_cm+'×'+x.width_cm)+'×'+v(x.height_cm)+'</td><td>'+v(x.envelope_l)+'</td><td>'+esc(x.confidence)+' · '+esc(x.measurement_status)+(x.association&&x.association.startsWith('ambiguous')?' · <span class="bf-warn">ambiguous</span>':'')+'</td></tr>';
 if(!ev.length)h+='<tr><td colspan="8">No new bags yet</td></tr>';return h+'</tbody></table></div>';}
function form(id,p){const cm=m=>m==null?'':Math.round(m*1000)/10;const f=[['camera_to_empty_floor_cm','camera→empty floor cm',cm(p.camera_to_empty_floor_m)],['usable_height_cm','usable floor→rim cm',cm(p.usable_height_m)],['tilt_from_vertical_deg','tilt °',p.tilt_from_vertical_deg??''],['capacity_l','capacity L',p.capacity_l]];
 return '<form data-cam="'+id+'"><b>'+name(id)+'</b> '+f.map(([k,l,val])=>'<label>'+l+' <input name="'+k+'" data-orig="'+val+'" value="'+val+'"></label>').join('')+'<button type="submit">Save</button></form>';}
let expectSession=null;
async function load(){try{const d=await (await fetch('/api/bin-fill',{cache:'no-store'})).json();if(d.error)throw new Error(d.error);const s=d.deposits;
 if(expectSession&&s.session_id!==expectSession)return;   // a response started before the reset
 for(const id of ['realsense','logitech'])$('bf-cam-'+id).innerHTML=card(id,d.cameras[id]||{},(s.cameras||{})[id]);
 $('bf-session').textContent='Session started '+new Date(s.session_started_at*1000).toLocaleString()+(s.resumed?' (resumed)':'')+' · bags present at start count toward fill, not as new';
}catch(err){$('bf-session').textContent='Bin fill data unavailable: '+err.message;}}

$('bf-reset').addEventListener('click',async()=>{if(!confirm('Reset? Both bag counts return to 0 and recording starts fresh (earlier rows stay in the CSV; calibration is kept).'))return;const res=await fetch('/api/bin-fill/session/new',{method:'POST',headers:bfHeaders()});const r=await res.json().catch(()=>({}));if(!res.ok){alert('Reset failed: '+(r.error||res.status));return;}expectSession=(r.session||{}).session_id||null;load();});
load();setInterval(load,1000);
})();
</script>
</section>
{% endraw %}"""
