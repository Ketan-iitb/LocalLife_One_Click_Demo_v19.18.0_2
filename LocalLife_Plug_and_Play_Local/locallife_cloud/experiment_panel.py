"""Research-page panel for the per-attempt experiment log (experiment_log.py).

Every attempt is recorded, including failures; the evaluation shown is over all
attempts of the run, with accuracy over valid measurements only and the failure
rate beside it. Wrapped in {% raw %} for the Jinja page.
"""

EXPERIMENT_PANEL = r"""{% raw %}
<section class="comparison" id="xp-panel"><h2>Experiment trials (every attempt logged)</h2>
<div class="muted">One trial = one reference object placed by you. Each camera's result within the time limit is logged: a
measurement, or why there was none (no detection, missing depth, missing calibration, unstable, rejected geometry,
stale reading, timeout, several objects). Calibration objects must differ from validation objects. While a run is
active, calibration actions are refused so the configuration stays frozen.</div>
<div class="calibrate"><input id="xp-run-name" placeholder="run name (e.g. validation day 1)">
<button onclick="xpRun('start')">Start evaluation run</button><button onclick="xpRun('stop')">Stop run</button>
<span id="xp-run" class="muted"></span></div>
<div class="calibrate">
<select id="xp-camera"><option value="both">both cameras</option><option value="realsense">RealSense</option><option value="logitech">Logitech</option></select>
<select id="xp-role"><option value="validation">validation object</option><option value="calibration">calibration object</option></select>
<input id="xp-object" placeholder="object id (unique per physical object)"><input id="xp-type" placeholder="object type">
<input id="xp-l" placeholder="L mm" size="5"><input id="xp-w" placeholder="W mm" size="5"><input id="xp-h" placeholder="H mm" size="5">
<input id="xp-ref" placeholder="reference volume L"><input id="xp-def" placeholder="volume definition (e.g. external L x W x H)">
<input id="xp-src" placeholder="reference source (ruler, water displacement)">
<input id="xp-colour" placeholder="true colour"><input id="xp-material" placeholder="true material">
<input id="xp-count" placeholder="objects in view" size="5"><input id="xp-timeout" placeholder="timeout s (5)" size="5">
<button onclick="xpAttempt()">Record attempt now</button></div>
<div id="xp-eval" class="small muted"></div>
<div class="scroll"><table><thead><tr><th>Camera</th><th>Role</th><th>Object</th><th>Outcome</th><th>Reported L</th><th>Raw L</th><th>Ref L</th><th>Colour</th><th>Material</th><th>Reason</th></tr></thead><tbody id="xp-rows"></tbody></table></div>
<div class="actions"><a class="export" href="/api/experiment/attempts.csv">Download all attempts (CSV)</a></div>
</section>
<script>
(function(){
const $=id=>document.getElementById(id);const v=id=>($(id).value||'').trim();
const num=x=>x===''?null:Number(x);const L=x=>x==null?'—':Number(x).toFixed(3);
const hdr=()=>{const h={'Content-Type':'application/json'};try{if(typeof API_TOKEN!=='undefined'&&API_TOKEN)h['X-API-Token']=API_TOKEN}catch(e){}return h};
window.xpRun=async function(action){const r=await fetch('/api/experiment/run/'+action,{method:'POST',headers:hdr(),body:JSON.stringify({name:v('xp-run-name')})});const d=await r.json();if(!r.ok)alert(d.error||'failed');refresh()};
window.xpAttempt=async function(){const body={camera:v('xp-camera'),role:v('xp-role'),object_id:v('xp-object'),object_type:v('xp-type'),
 reference_dims_mm:{length:num(v('xp-l')),width:num(v('xp-w')),height:num(v('xp-h'))},reference_volume_l:num(v('xp-ref')),
 reference_volume_definition:v('xp-def'),reference_source:v('xp-src'),actual_colour:v('xp-colour')||null,actual_material:v('xp-material')||null,
 expected_count:num(v('xp-count')),timeout_s:num(v('xp-timeout'))||5};
 if(!body.object_id){alert('Enter an object id');return}
 $('xp-eval').textContent='Measuring… (waits for a newly processed frame)';
 const r=await fetch('/api/experiment/attempt',{method:'POST',headers:hdr(),body:JSON.stringify(body)});const d=await r.json();if(!r.ok)alert(d.error||'failed');refresh()};
function row(a){const tr=document.createElement('tr');for(const x of [a.camera,a.role,a.object_id,a.outcome,L(a.reported_volume_l)+(a.template_volume_used?' (template)':''),L(a.raw_geometric_volume_l),L(a.reference_volume_l),a.colour||'—',a.material||'—',a.reason||'']){const td=document.createElement('td');td.textContent=String(x);tr.appendChild(td)}return tr}
async function refresh(){try{const runs=await (await fetch('/api/experiment/runs')).json();const active=runs.active;
 $('xp-run').textContent=active?'Active run '+active+' — calibration frozen':'No run active';
 const q=active?'?run_id='+active:'';const a=await (await fetch('/api/experiment/attempts'+q)).json();
 $('xp-rows').replaceChildren(...(a.attempts||[]).slice(-25).reverse().map(row));
 const e=await (await fetch('/api/experiment/evaluation'+q)).json();
 const parts=Object.entries(e.cameras||{}).map(([c,m])=>c+': '+m.valid_measurements+'/'+m.attempts+' valid (failure '+(m.failure_rate==null?'—':Math.round(m.failure_rate*100)+'%')+')'+(m.volume.n?', MAE '+L(m.volume.mae_l)+' L, RMSE '+L(m.volume.rmse_l)+', bias '+L(m.volume.bias_l)+', MAPE '+m.volume.mape_percent.toFixed(1)+'%':', no valid volume with a reference yet'));
 $('xp-eval').textContent=(parts.join(' · ')||'No validation attempts yet.')+(e.objects_used_for_both_calibration_and_validation&&e.objects_used_for_both_calibration_and_validation.length?' · EXCLUDED (used for calibration too): '+e.objects_used_for_both_calibration_and_validation.join(', '):'')+(e.configuration_consistent===false?' · WARNING: configuration changed during the run':'');
}catch(err){}}
refresh();setInterval(refresh,5000);
})();
</script>
{% endraw %}"""
