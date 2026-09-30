"""v45 "Bin fill & new deposits" panel for / and /research (renders /api/bin-fill only).

Wrapped in {% raw %} because both pages are Jinja templates.
"""

BIN_FILL_PANEL = r"""{% raw %}
<section id="bf-panel" style="margin:14px 0;padding:16px;border:1px solid rgba(128,160,170,.35);border-radius:14px;background:rgba(17,29,35,.55)">
<style>
#bf-panel{font-size:13px}#bf-panel h2{margin:0 0 4px;font-size:18px}#bf-panel .bf-muted{opacity:.72}
#bf-panel .bf-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:8px}
#bf-panel .bf-card{border:1px solid rgba(128,160,170,.3);border-radius:10px;padding:10px}
#bf-panel .bf-big{font-size:26px;font-weight:800}#bf-panel .bf-na{color:#e0795c;font-weight:700}
#bf-panel table{width:100%;border-collapse:collapse;margin-top:8px}#bf-panel th,#bf-panel td{padding:5px 6px;border-bottom:1px solid rgba(128,160,170,.22);text-align:left;vertical-align:top}
#bf-panel th{font-size:12px;opacity:.8}#bf-panel details{margin-top:6px}
#bf-panel input,#bf-panel select{background:transparent;color:inherit;border:1px solid rgba(128,160,170,.5);border-radius:7px;padding:4px;width:90px}
#bf-panel select option{color:#111}#bf-panel button,#bf-panel a.bf-btn{border:1px solid rgba(128,160,170,.6);border-radius:8px;padding:6px 10px;color:inherit;background:rgba(80,120,130,.25);cursor:pointer;font-weight:700;text-decoration:none}
@media(max-width:900px){#bf-panel .bf-grid{grid-template-columns:1fr}}
</style>
<h2>Bin fill &amp; new deposits</h2>
<div class="bf-muted" id="bf-session">Loading…</div>
<div class="bf-grid"><div class="bf-card"><div class="bf-muted">NEW bags this session</div><div class="bf-big" id="bf-count">0</div><div class="bf-muted" id="bf-rej"></div></div>
<div class="bf-card" id="bf-cam-realsense"></div><div class="bf-card" id="bf-cam-logitech"></div></div>
<table><thead><tr><th>#</th><th>Event</th><th>Deposited</th><th>Cameras</th><th>Bag</th><th>Colour</th><th>Type / material</th><th>L×W×H cm</th><th>Envelope L</th><th>Occupancy change L</th><th>Status</th></tr></thead><tbody id="bf-rows"></tbody></table>
<div class="bf-muted" id="bf-labels"></div>
<a class="bf-btn" href="/api/session-deposits.csv">Download deposits CSV</a>
<script>
(function(){
const $=id=>document.getElementById(id);const t=s=>s?new Date(s*1000).toLocaleTimeString():'—';const v=(x,u)=>x==null?'N/A':x+(u||'');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function cam(id,r){const p=r.profile||{};let h='<b>'+(id==='realsense'?'RealSense':'Logitech')+'</b> ';
 if(r.status!=='ok'){h+='<div class="bf-na">Fill: N/A</div><div>'+esc(r.reason)+'</div>';}
 else{h+=(r.stale?'<span class="bf-na">(stale: scene moving)</span>':'')+'<div class="bf-big">'+r.height_fill_pct+'% <span class="bf-muted" style="font-size:13px">height fill</span></div>'+
  '<div>Max reliable fill height: <b>'+r.max_fill_height_cm+' cm</b> of '+r.usable_height_cm+' cm · remaining '+r.remaining_height_cm+' cm</div>'+
  '<div>≈ '+r.rough_litres+' L used, ≈ '+r.rough_remaining_litres+' L left <span class="bf-muted">('+esc(r.rough_litres_label)+'; '+esc(r.capacity_note)+')</span></div>'+
  '<div>Estimated occupied volume: '+(r.occupied_l==null?'N/A — '+esc(r.occupied_reason):r.occupied_l+' L ('+r.occupied_pct+'%)')+'</div>'+
  '<div class="bf-muted">depth coverage '+r.coverage_pct+'%</div>';}
 h+='<div class="bf-muted">Last updated: '+t(r.updated_at)+(r.measurement_zone?'':' · measurement zone not drawn')+'</div>';
 h+='<details><summary>Fill profile (this camera only)</summary><form data-cam="'+id+'">'+
 [['camera_to_empty_floor_cm','cam→empty floor cm'],['tilt_from_vertical_deg','tilt from vertical °'],['usable_height_cm','usable floor→rim cm'],['camera_above_rim_cm','camera above rim cm'],['inner_length_cm','inner length cm'],['inner_width_cm','inner width cm'],['capacity_l','capacity L']].map(([k,l])=>{
  const src={camera_to_empty_floor_cm:p.camera_to_empty_floor_m,usable_height_cm:p.usable_height_m,camera_above_rim_cm:p.camera_above_rim_m,inner_length_cm:p.inner_length_m,inner_width_cm:p.inner_width_m}[k];
  const val=k.endsWith('_cm')?(src==null?'':Math.round(src*1000)/10):(p[k]??'');return '<label>'+l+' <input name="'+k+'" value="'+val+'"></label> ';}).join('')+
 '<label>distance is <select name="distance_kind">'+['unknown','vertical','optical_axis'].map(o=>'<option'+(p.distance_kind===o?' selected':'')+'>'+o+'</option>').join('')+'</select></label> '+
 '<label><input type="checkbox" name="capacity_verified" style="width:auto"'+(p.capacity_verified?' checked':'')+'> capacity verified on bin label</label> <button type="submit">Save (bin empty, tripod fixed)</button>'+
 '<div class="bf-muted">'+esc((r.profile_problems||[]).join('; '))+'</div></form></details>';return h;}
async function load(){try{const d=await (await fetch('/api/bin-fill')).json();const s=d.deposits;
 for(const id of ['realsense','logitech']){const box=$('bf-cam-'+id);if(box.querySelector('details[open]'))continue;box.innerHTML=cam(id,d.cameras[id]||{});}
 $('bf-count').textContent=s.new_bags_this_session;$('bf-rej').textContent=(s.warmup?'Start-up warm-up: bags in view become the baseline. ':'')+s.rejected_candidates+' candidate(s) not counted (old bag moved / re-detected / no rise)';
 $('bf-session').textContent='Session started '+new Date(s.session_started_at*1000).toLocaleString()+' · last update '+t(s.updated_at)+' · existing bags at start are part of the fill, not new';
 $('bf-labels').textContent='Envelope = '+s.envelope_label+'. Occupancy change = '+s.delta_label+'. Neither is the height-based litres figure.';
 const rows=$('bf-rows');rows.replaceChildren();s.events.forEach((e,i)=>{const tr=document.createElement('tr');
  [i+1,e.event_id,new Date(e.deposit_time*1000).toLocaleTimeString(),e.cameras.join(' + '),e.track_id??'—',e.colour,e.type+' / '+e.material,
   (e.length_cm==null?'N/A':e.length_cm+'×'+e.width_cm+'×'+v(e.height_cm)),v(e.envelope_l),v(e.delta_occupancy_l),e.measurement_status+(e.reason?' — '+e.reason:'')].forEach(x=>{const td=document.createElement('td');td.textContent=String(x);tr.appendChild(td);});rows.appendChild(tr);});
 if(!s.events.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=11;td.textContent='No new bags yet this session';tr.appendChild(td);rows.appendChild(tr);}
}catch(err){$('bf-session').textContent='Bin fill data unavailable';}}
document.addEventListener('submit',async ev=>{const f=ev.target;if(!f.dataset||!f.dataset.cam)return;ev.preventDefault();const body={};
 for(const el of f.elements){if(!el.name)continue;body[el.name]=el.type==='checkbox'?el.checked:el.value;}
 const r=await fetch('/api/cameras/'+f.dataset.cam+'/fill-profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();alert(r.ok?('Saved. '+((j.problems||[]).length?'Problems: '+j.problems.join('; '):'Profile complete.')):('Not saved: '+(j.error||r.status)));f.closest('details').open=false;load();});
load();setInterval(load,2000);
})();
</script>
</section>
{% endraw %}"""
