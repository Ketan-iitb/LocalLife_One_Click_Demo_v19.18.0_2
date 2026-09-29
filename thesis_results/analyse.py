import csv, collections, statistics as st
OUT="."
# ---------- 1. Phase 1 ground-truth table (from phase_1_local_life_testing.pdf) ----------
# name, actual LxWxH cm, actual material, actual colour, measured LxWxH cm, measured L, measured material, material conf %, measured colour
GT=[("Shoes box",(36,25,12.5),"paper","black",(36,24,12.5),9.8,"cardboard",100,"black"),
("Black polythene bag",(30,30,20),"plastic","black",(28,20,26),12,"polythene bag",100,"black"),
("Apple waste bag",(25,20,8),"plastic","transparent",(24,16,10),2.6,"polythene bag",100,"grey"),
("Backpack",(35,12,40),"fabric","black",(42,42,13),14,"cardboard",75,"black"),
("Polythene bag",(26,24,12),"plastic","white",(25,22,10),1.4,"polythene bag",100,"grey"),
("Toy",(12,10,20),"fabric","grey",(15,9,20),1.7,"metal",100,"orange"),
("Brown paper bag",(13,30,30),"paper","brown",(10,28,29),28,"mixed waste",100,"brown"),
("Drink can",(17,7,7),"metal","pink",(17,6,7),0.5,"fabric",50,"red"),
("Milk box",(9.5,7,22),"paper","white",(8.2,6,23.5),1.1,"plastic",100,"blue"),
("Tea box",(11,11,17),"plastic","orange",(11,8,17),0.8,"plastic",100,"orange"),
("Small plastic box",(3.5,3.5,5),"plastic","bluish green",(4,3,6),0.07,"plastic",50,"cyan"),
("Plastic cream bottle",(8,4,20),"plastic","white",(8,3,20),0.3,"plastic",100,"red"),
("Gym equipment",(15,12,1.5),"plastic","green",(12,9,1.4),0.08,"metal",50,"green"),
("Food waste bag",(20,19,4),"plastic","yellow",(23,20,5),0.8,"food waste",100,"orange"),
("Pillow",(40,40,12),"fabric","grey",(43,38,12),18,"fabric",100,"brown")]
MAT_OK={("paper","cardboard"),("plastic","polythene bag"),("plastic","plastic"),("fabric","fabric"),("metal","metal")}
COL_NEAR={("transparent","grey"),("white","grey"),("pink","red"),("bluish green","cyan"),("yellow","orange")}
rows=[];ax_err=[];vol_err=[];mat_ok=col_ok=col_near=0
for n,a,am,ac,m,mv,mm,mc,mcol in GT:
    sa,sm=sorted(a,reverse=True),sorted(m,reverse=True)          # orientation-independent
    e=[abs(x-y)/y*100 for x,y in zip(sm,sa)]; ax_err+=e
    env=a[0]*a[1]*a[2]/1000; menv=m[0]*m[1]*m[2]/1000
    ve=(mv-env)/env*100; vol_err.append(abs(ve))
    mo=(am,mm) in MAT_OK; mat_ok+=mo
    co=ac==mcol; cn=(ac,mcol) in COL_NEAR; col_ok+=co; col_near+=cn
    flag=[]
    if mv>1.2*menv: flag.append("reported litres > its own LxWxH envelope")
    if not mo and mc>=75: flag.append(f"wrong material shown at {mc}%")
    rows.append([n,"x".join(map(str,a)),round(env,2),"x".join(map(str,m)),mv,round(menv,2),
                 *[round(x,1) for x in e],round(ve,1),am,mm,mc,"yes" if mo else "no",ac,mcol,
                 "yes" if co else ("close" if cn else "no"),"; ".join(flag)])
with open(f"{OUT}/phase1_ground_truth_errors.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["object","actual_LxWxH_cm","actual_envelope_L","measured_LxWxH_cm","reported_volume_L",
        "measured_envelope_L","err_longest_%","err_middle_%","err_shortest_%","volume_err_vs_envelope_%",
        "actual_material","measured_material","material_conf_%","material_correct","actual_colour","measured_colour","colour_correct","flags"]); w.writerows(rows)
gt_summary=dict(n=len(GT),dim_median=round(st.median(ax_err),1),dim_within10=round(100*sum(x<=10 for x in ax_err)/len(ax_err)),
   dim_within20=round(100*sum(x<=20 for x in ax_err)/len(ax_err)),vol_median=round(st.median(vol_err),1),
   vol_within25=sum(x<=25 for x in vol_err),mat=mat_ok,col=col_ok,col_near=col_near)
# ---------- 2. comparison_measurements CSV: flag and summarise ----------
import sys
F=sys.argv[1] if len(sys.argv)>1 else "comparison_measurements_flagged.csv"  # usage: python3 analyse.py path/to/comparison_measurements.csv
R=list(csv.DictReader(open(F)))
f_=lambda x: float(x) if x not in ("",None) else None
ev=collections.defaultdict(dict)
for r in R: ev[r["comparison_event_id"]][r["camera_source"]]=r
FAM={"can":"can","bottle":"bottle","carton":"carton","box":"box","bag":"bag","backpack":"bag","rucksack":"bag","handbag":"bag","slipper":"shoe","shoe":"shoe"}
fam=lambda t: next((v for k,v in FAM.items() if k in (t or "").lower()),(t or "").lower())
for r in R:
    fl=[]
    if r["status"]!="accepted": fl.append(r["status"])
    L,W,H,V=(f_(r[k]) for k in ("length_mm","width_mm","height_mm","selected_volume_litres"))
    if r["status"]=="accepted" and ((L or 0)>1000 or (W or 0)>1000 or (H or 0)>700 or (V or 0)>60): fl.append("implausible_size_outlier")
    other=ev[r["comparison_event_id"]].get("logitech" if r["camera_source"]=="realsense" else "realsense")
    if r["status"]=="accepted" and other and other["status"]=="accepted" and fam(other["object_type"])!=fam(r["object_type"]):
        fl.append("paired_with_different_object")
    r["quality_flag"]=";".join(fl) or "usable"
with open(f"{OUT}/comparison_measurements_flagged.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(R)
cs={}
for cam in ("realsense","logitech"):
    x=[r for r in R if r["camera_source"]==cam]; c=collections.Counter(r["status"] for r in x)
    use=[r for r in x if r["quality_flag"]=="usable"]
    cs[cam]=dict(n=len(x),acc=c["accepted"],rej=c["rejected"],miss=c["missing"],outl=sum("implausible" in r["quality_flag"] for r in x),
        usable=len(use),reasons=collections.Counter(r["reason"] for r in x if r["status"]=="rejected").most_common(3))
pairs=[e for e in ev.values() if all(k in e and e[k]["status"]=="accepted" for k in ("realsense","logitech"))]
same=[e for e in pairs if "paired_with_different_object" not in e["realsense"]["quality_flag"]]
# ---------- 3. summary markdown ----------
s=gt_summary
md=f"""# Preliminary results (auto-generated by analyse.py)

## A. Phase 1 ground-truth test ({s['n']} objects, tape-measured; source: phase_1_local_life_testing.pdf)
Dimensions compared largest-to-largest (orientation-independent). "Envelope" = L x W x H of the real object.

| Metric | Result |
|---|---|
| Median dimension error | {s['dim_median']} % |
| Dimensions within 10 % / 20 % | {s['dim_within10']} % / {s['dim_within20']} % of {3*s['n']} |
| Median volume error vs actual envelope | {s['vol_median']} % (only {s['vol_within25']}/{s['n']} within 25 %) |
| Material correct | {s['mat']}/{s['n']} |
| Colour exact / close | {s['col']}/{s['n']} exact, +{s['col_near']} close (e.g. white->grey, pink->red) |

Note: reported litres are the measured (height-map) volume, not a box envelope, so volume "error" mixes shape and measurement error. Which camera produced this table must be stated in the thesis.

## B. Automatic dual-camera log (comparison_measurements.csv, 22-29 Sep, {len(R)} rows, no ground truth)

| | RealSense | Logitech |
|---|---|---|
| Attempts | {cs['realsense']['n']} | {cs['logitech']['n']} |
| Accepted | {cs['realsense']['acc']} ({round(100*cs['realsense']['acc']/cs['realsense']['n'])} %) | {cs['logitech']['acc']} ({round(100*cs['logitech']['acc']/cs['logitech']['n'])} %) |
| Rejected with reason | {cs['realsense']['rej']} | {cs['logitech']['rej']} |
| Missing (no final reading) | {cs['realsense']['miss']} | {cs['logitech']['miss']} |
| Implausible-size outliers | {cs['realsense']['outl']} | {cs['logitech']['outl']} |
| Usable after flagging | {cs['realsense']['usable']} | {cs['logitech']['usable']} |

Paired events where both cameras accepted: {len(pairs)}; of these only {len(same)} are the same object type in both cameras
(the rest were paired with a different object by the time window) -> no cross-camera accuracy claim from this log.
Top rejection reasons: RealSense {cs['realsense']['reasons']}; Logitech {cs['logitech']['reasons']}.
"""
open(f"{OUT}/preliminary_results.md","w").write(md); print(md)
