"""v45 "Bin fill & new deposits" panel for / and /research (renders /api/bin-fill only).

State lives in the backend: a page refresh never resets the session or the count.
Wrapped in {% raw %} because both pages are Jinja templates.
"""

BIN_FILL_PANEL = r"""{% raw %}
<section id="bf-panel" style="margin:14px 0;padding:16px;border:1px solid rgba(128,160,170,.35);border-radius:14px;background:rgba(17,29,35,.55)">
<style>
#bf-panel{font-size:13px}#bf-panel h2{margin:0 0 4px;font-size:18px}#bf-panel .bf-muted{opacity:.72}
#bf-panel .bf-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:8px}
#bf-panel .bf-card{border:1px solid rgba(128,160,170,.3);border-radius:10px;padding:10px}
#bf-panel .bf-big{font-size:26px;font-weight:800}#bf-panel .bf-na{color:#e0795c;font-weight:700}#bf-panel .bf-ok{color:#3fbf8a;font-weight:700}
#bf-panel .bf-warn{color:#e0b85c}
#bf-panel table{width:100%;border-collapse:collapse;margin-top:8px}#bf-panel th,#bf-panel td{padding:5px 6px;border-bottom:1px solid rgba(128,160,170,.22);text-align:left;vertical-align:top}
#bf-panel th{font-size:12px;opacity:.8}#bf-panel details{margin-top:8px}
#bf-panel input,#bf-panel select{background:transparent;color:inherit;border:1px solid rgba(128,160,170,.5);border-radius:7px;padding:4px;width:80px}
#bf-panel select option{color:#111}#bf-panel button,#bf-panel a.bf-btn{border:1px solid rgba(128,160,170,.6);border-radius:8px;padding:6px 10px;color:inherit;background:rgba(80,120,130,.25);cursor:pointer;font-weight:700;text-decoration:none}
#bf-panel .bf-set{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}#bf-panel .bf-set label{display:inline-block;margin:3px 8px 3px 0}
@media(max-width:900px){#bf-panel .bf-grid,#bf-panel .bf-set{grid-template-columns:1fr}}
</style>
<h2>Bin fill &amp; new deposits</h2>
<div id="bf-session" class="bf-muted">Loading…</div>
<div class="bf-grid">
 <div class="bf-card"><div class="bf-muted">NEW bags this session</div><div class="bf-big" id="bf-count">0</div>
  <div id="bf-state"></div><div class="bf-muted" id="bf-last"></div><div class="bf-muted" id="bf-rej"></div></div>
 <div class="bf-card" id="bf-cam-realsense"></div><div class="bf-card" id="bf-cam-logitech"></div></div>
<table><thead><tr><th>#</th><th>Event</th><th>Deposited</th><th>Colour</th><th>Type / material</th><th>L×W×H cm</th><th>Envelope L</th><th>Occupancy Δ L</th><th>Evidence / status</th></tr></thead><tbody id="bf-rows"></tbody></table>
<div class="bf-muted" id="bf-labels"></div>
<div style="margin-top:8px"><a class="bf-btn" href="/api/session-deposits.csv">Download deposits CSV</a> <button type="button" id="bf-new">Start new deposit session</button></div>
<details id="bf-settings"><summary>Installation settings (optional corrections, per camera)</summary><div class="bf-set" id="bf-forms"></div>
<div class="bf-muted">Only changed fields are saved. Saved values survive restarts and always take precedence over the provisional defaults. Tilt left blank = not measured (reading stays "approximate").</div></details>
<script>
(function(){
const $=id=>document.getElementById(id);const t=s=>s?new Date(s*1000).toLocaleTimeString():'—';const v=x=>x==null?'N/A':x;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const name=id=>id==='realsense'?'RealSense':'Logitech';const dur=s=>{s=Math.round(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return (h?h+' h ':'')+m+' min '+(s%60)+' s';};
const STATE={initialising:'Initialising baseline…',watching:'Watching for deposits',candidate:'Candidate: something is entering',settling:'Settling…'};
function cam(id,r,w){let h='<b>'+name(id)+'</b> <span class="bf-muted">('+esc(r.profile_status||'')+' profile)</span>';
 if(r.status!=='ok'){h+='<div class="bf-na">Fill: unavailable</div><div>'+esc(r.reason)+'</div>';}
 else{h+=(r.stale?' <span class="bf-warn">(scene moving – last settled reading)</span>':'')+
  '<div>Max reliable waste height: <b>'+r.max_fill_height_cm+' cm</b> above floor · remaining '+r.remaining_height_cm+' cm</div>'+
  '<div class="bf-big">'+r.height_fill_pct+'% <span class="bf-muted" style="font-size:13px">HEIGHT fill (of '+r.usable_height_cm+' cm)</span></div>'+
  '<div>≈ '+r.rough_litres+' L (≈ '+r.rough_remaining_litres+' L left) <span class="bf-muted">– height-based approximation; assumes roughly uniform filling; '+esc(r.capacity_note)+'</span></div>'+
  '<div>Estimated occupied volume: '+(r.occupied_l==null?'<span class="bf-muted">N/A – '+esc(r.occupied_reason)+'</span>':r.occupied_l+' L ('+r.occupied_pct+'%)')+'</div>';
  for(const x of r.warnings||[])h+='<div class="bf-warn">⚠ '+esc(x)+'</div>';}
 h+='<div class="bf-muted">Last frame '+t(r.last_processed_at)+' · last valid '+t(r.last_valid_at)+'</div>';
 const p=r.profile||{};const cm=m=>m==null?'not measured':Math.round(m*100)+' cm';
 h+='<div class="bf-muted" title="'+esc((r.assumptions||[]).join('\n'))+'">Floor '+cm(p.camera_to_empty_floor_m)+' · usable '+cm(p.usable_height_m)+' · tilt '+(p.tilt_from_vertical_deg==null?'not measured':p.tilt_from_vertical_deg+'°')+' · '+esc(p.source||'')+((r.assumptions||[]).length?' ⓘ':'')+'</div>';
 h+='<div class="bf-muted">Deposit watcher: '+esc(w?(STATE[w.state]||w.state)+(w.reason?' – '+w.reason:''):'no frames yet')+' · event region: '+esc(r.event_region)+'</div>';return h;}
function form(id,p){const cm=m=>m==null?'':Math.round(m*1000)/10;const f=[['camera_to_empty_floor_cm','cam→empty floor cm',cm(p.camera_to_empty_floor_m)],['usable_height_cm','usable floor→rim cm',cm(p.usable_height_m)],['camera_above_rim_cm','camera above rim cm',cm(p.camera_above_rim_m)],['tilt_from_vertical_deg','tilt from vertical °',p.tilt_from_vertical_deg??''],['inner_length_cm','inner length cm',cm(p.inner_length_m)],['inner_width_cm','inner width cm',cm(p.inner_width_m)],['capacity_l','capacity L',p.capacity_l]];
 return '<form data-cam="'+id+'"><b>'+name(id)+'</b><br>'+f.map(([k,l,val])=>'<label>'+l+' <input name="'+k+'" data-orig="'+val+'" value="'+val+'"></label>').join('')+
 '<label>distance is <select name="distance_kind" data-orig="'+esc(p.distance_kind)+'">'+['unknown','vertical','optical_axis'].map(o=>'<option'+(p.distance_kind===o?' selected':'')+'>'+o+'</option>').join('')+'</select></label>'+
 '<label><input type="checkbox" name="capacity_verified" data-orig="'+(!!p.capacity_verified)+'" style="width:auto"'+(p.capacity_verified?' checked':'')+'> capacity checked on bin label</label> <button type="submit">Save '+name(id)+'</button></form>';}
async function load(){try{const d=await (await fetch('/api/bin-fill',{cache:'no-store'})).json();const s=d.deposits;
 for(const id of ['realsense','logitech'])$('bf-cam-'+id).innerHTML=cam(id,d.cameras[id]||{},(s.cameras||{})[id]);
 if(!$('bf-settings').open)$('bf-forms').innerHTML=['realsense','logitech'].map(id=>form(id,(d.cameras[id]||{}).profile||{})).join('');
 $('bf-count').textContent=s.new_bags_this_session;
 $('bf-state').innerHTML='<span class="'+(s.status==='watching'?'bf-ok':'bf-warn')+'">'+esc(STATE[s.status]||s.status)+'</span>';
 $('bf-last').textContent='Last confirmed deposit: '+t(s.last_confirmed_at);
 $('bf-rej').textContent=s.rejected_candidates+' candidate(s) rejected'+(s.recent_rejections.length?' – latest: '+s.recent_rejections[s.recent_rejections.length-1].reason:'');
 $('bf-session').textContent='Session started '+new Date(s.session_started_at*1000).toLocaleString()+' · running '+dur(s.elapsed_s)+' · last frame '+t(s.updated_at)+(s.resumed?' · resumed after restart':'')+' · bags present at start are part of the fill, not new';
 $('bf-labels').textContent='Envelope = '+s.envelope_label+'. Occupancy Δ = '+s.delta_label+'. Neither is the height-based litres figure.';
 const rows=$('bf-rows');rows.replaceChildren();s.events.forEach((e,i)=>{const tr=document.createElement('tr');const ev=Object.entries(e.evidence||{}).map(([c,x])=>name(c)+': '+x.evidence).join('; ');
  [i+1,e.event_id,new Date(e.deposit_time*1000).toLocaleTimeString(),e.colour,e.object_type+' ('+e.detector_label+') / '+e.material,
   (e.length_cm==null?'N/A':e.length_cm+'×'+e.width_cm)+'×'+v(e.height_cm),v(e.envelope_l),v(e.delta_occupancy_l),ev+' · '+e.association+' · '+e.measurement_status+(e.reason?' – '+e.reason:'')].forEach(x=>{const td=document.createElement('td');td.textContent=String(x);tr.appendChild(td);});rows.appendChild(tr);});
 if(!s.events.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=9;td.textContent='No new bags yet this session';tr.appendChild(td);rows.appendChild(tr);}
}catch(err){$('bf-session').textContent='Bin fill data unavailable';}}
document.addEventListener('submit',async ev=>{const f=ev.target;if(!f.dataset||!f.dataset.cam||!f.closest('#bf-panel'))return;ev.preventDefault();const body={};
 for(const el of f.elements){if(!el.name)continue;const val=el.type==='checkbox'?String(el.checked):el.value;if(val!==el.dataset.orig)body[el.name]=el.type==='checkbox'?el.checked:el.value;}
 if(!Object.keys(body).length){alert('Nothing changed.');return;}
 const r=await fetch('/api/cameras/'+f.dataset.cam+'/fill-profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();alert(r.ok?('Saved ('+j.status+').'+((j.blocking||[]).length?' Still missing: '+j.blocking.join('; '):'')):('Not saved: '+(j.error||r.status)));$('bf-settings').open=false;load();});
$('bf-new').addEventListener('click',async()=>{if(!confirm('Start a new deposit session? The NEW-bag count returns to 0 and the baseline is re-taken.'))return;await fetch('/api/bin-fill/session/new',{method:'POST'});load();});
load();setInterval(load,2000);
})();
</script>
</section>
{% endraw %}"""
