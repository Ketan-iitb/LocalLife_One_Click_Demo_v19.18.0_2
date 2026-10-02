"""Research page layout: the SAME content, grouped and ordered so it reads top-down.

Nothing is removed or renamed. After load, existing blocks are moved (DOM nodes keep their ids, so
every existing script keeps writing into them) into this order:

  1. section menu
  2. both camera views with their key readouts (preview, depth view, live detections, history);
     per camera, colours / materials / before-after occupancy go into one "More details" fold
  3. Bin fill & new deposits
  4. Local vs Cloud comparison (table and graphs, open)
  5. Local vs Cloud: Cost and Accuracy (open)
  6. Camera comparison (matched objects, paired events) -- one fold
  7. Setup guide, fused and recipe results -- one fold

Folds remember open/closed per browser. Wrapped in {% raw %} for the Jinja page.
"""

RESEARCH_LAYOUT = r"""{% raw %}
<style>
#rl-nav{position:sticky;top:0;z-index:20;display:flex;gap:6px;flex-wrap:wrap;padding:8px 10px;margin:0 0 12px;border:1px solid var(--line,#27404a);border-radius:12px;background:rgba(8,17,22,.94)}
#rl-nav a{color:inherit;text-decoration:none;font-weight:650;font-size:13px;padding:5px 10px;border-radius:8px;background:rgba(80,120,130,.22)}
#rl-nav a:hover{background:rgba(80,120,130,.4)}
details.rl-fold{margin:12px 0;border:1px solid var(--line,#27404a);border-radius:12px;padding:6px 12px;background:rgba(16,29,37,.6)}
details.rl-fold>summary{cursor:pointer;font-weight:750;font-size:15px;padding:4px 0}
details.rl-more{margin-top:10px;border-top:1px dashed var(--line,#27404a);padding-top:6px}
details.rl-more>summary{cursor:pointer;font-weight:650}
.rl-anchor{scroll-margin-top:60px}
</style>
<script>
(function(){
function store(key,open){try{localStorage.setItem('rl:'+key,open?'1':'0')}catch(e){}}
function remembered(key,dflt){try{const v=localStorage.getItem('rl:'+key);return v==null?dflt:v==='1'}catch(e){return dflt}}
function fold(key,title,cls,open){const d=document.createElement('details');d.className=cls;d.dataset.rl=key;d.open=remembered(key,open);
 const s=document.createElement('summary');s.textContent=title;d.appendChild(s);d.addEventListener('toggle',()=>store(key,d.open));return d;}
function organise(){
 const main=document.querySelector('main')||document.body;
 if(main.querySelector('#rl-nav'))return;
 const cams=main.querySelector('section.columns');
 // 2. per camera: colours, materials, before/after occupancy -> one "More details" fold
 for(const cam of (cams?cams.querySelectorAll(':scope > *'):[])){
  const heads=[...cam.querySelectorAll(':scope > h3')].filter(h=>/colou?rs$|materials$|bin occupancy/i.test(h.textContent.trim()));
  if(!heads.length)continue;
  const name=(cam.querySelector('h2')||{}).textContent||'camera';
  const more=fold('more-'+name.trim().split(/\s+/)[1],'More details — colours, materials, before/after occupancy','rl-more',false);
  for(const h of heads){const group=[h];let n=h.nextElementSibling;while(n&&!/^(H2|H3)$/.test(n.tagName)&&!(n.tagName==='DETAILS'&&n.classList.contains('wizard'))){group.push(n);n=n.nextElementSibling;}group.forEach(el=>more.appendChild(el));}
  cam.appendChild(more);
 }
 // 6. camera comparison sections -> one fold after the cost panel
 const comps=[...main.querySelectorAll(':scope > section.comparison')];
 let compFold=null;
 if(comps.length){compFold=fold('camera-comparison','Camera comparison — matched objects and paired events (RealSense vs Logitech)','rl-fold rl-anchor',false);compFold.id='rl-camera-comparison';comps.forEach(s=>compFold.appendChild(s));}
 // 7. setup guide + fused + recipe results -> one fold at the end
 const extra=[...main.querySelectorAll(':scope > section.setup, :scope > section.fused')];
 let setupFold=null;
 if(extra.length){setupFold=fold('setup-results','Setup guide, fused result and recipe result','rl-fold rl-anchor',false);setupFold.id='rl-setup';extra.forEach(s=>setupFold.appendChild(s));}
 // order after the cameras: bin fill, local vs cloud, cost & accuracy, camera comparison, setup
 let anchor=cams;
 for(const el of [main.querySelector('#bf-panel'),main.querySelector('#lc-panel'),main.querySelector('#ca-panel'),compFold,setupFold]){
  if(!el||!anchor)continue;anchor.after(el);anchor=el;}
 if(cams){cams.id=cams.id||'rl-cameras';cams.classList.add('rl-anchor');}
 for(const id of ['bf-panel','lc-panel','ca-panel']){const el=main.querySelector('#'+id);if(el)el.classList.add('rl-anchor');}
 // 1. section menu
 const nav=document.createElement('nav');nav.id='rl-nav';
 const links=[[cams&&cams.id,'Cameras'],['bf-panel','Bin fill & deposits'],['lc-panel','Local vs Cloud'],['ca-panel','Cost & accuracy'],[compFold&&compFold.id,'Camera comparison'],[setupFold&&setupFold.id,'Setup & results']];
 nav.innerHTML=links.filter(([id])=>id&&document.getElementById(id)).map(([id,t])=>'<a href="#'+id+'">'+t+'</a>').join('');
 nav.addEventListener('click',e=>{const a=e.target.closest('a');if(!a)return;const t=document.getElementById(a.getAttribute('href').slice(1));if(t&&t.tagName==='DETAILS')t.open=true;});
 (cams||main.firstElementChild).before(nav);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',organise);else organise();
})();
</script>
{% endraw %}"""
