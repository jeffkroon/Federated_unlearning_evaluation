import csv, os
FIN=os.environ.get("FIN","THESIS_RESULTS_FINAL")
OUT=f"{FIN}/latex"
DS_ORDER=["mnist","adult","cifar10"]; DS_NAME={"mnist":"MNIST","adult":"Adult","cifar10":"CIFAR-10"}
ST_ORDER=["exact_retraining","federated_exact_retraining","sisa","distillation","mf"]
ST_NAME={"exact_retraining":"Exact RT","federated_exact_retraining":"Fed.\\ Exact RT","sisa":"SISA","distillation":"Distillation","mf":"MF"}

def load(path):
    d={}
    for r in csv.DictReader(open(path)):
        d[(r["dataset"],r["strategy"])]=r
    return d

agg=load(f"{FIN}/aggregated_mean_std.csv")
s4=load(f"{FIN}/aggregated_s4_temporary.csv")

def cell(r, base):
    m=r.get(base+"_mean",""); s=r.get(base+"_std","")
    if m=="" or m is None: return "---"
    m=float(m)
    if int(r["n_seeds"])>1 and s not in("",None):
        return f"${m:.3f} \\pm {float(s):.3f}$"
    return f"${m:.3f}$"

def table(fname, caption, label, metrics, headers, source=agg, note=""):
    cols="l l "+" ".join("c"*len(metrics))
    L=[r"\begin{table}[H]",r"\centering",f"\\caption{{{caption}}}",f"\\label{{{label}}}",
       r"\small",r"\renewcommand{\arraystretch}{1.2}",f"\\begin{{tabular}}{{@{{}} {cols} @{{}}}}",r"\toprule",
       "\\textbf{Dataset} & \\textbf{Strategy} & "+" & ".join(f"\\textbf{{{h}}}" for h in headers)+r" \\",r"\midrule"]
    for di,ds in enumerate(DS_ORDER):
        for si,strat in enumerate(ST_ORDER):
            r=source.get((ds,strat))
            if not r: continue
            name=DS_NAME[ds] if si==0 else ""
            pre=f"\\multirow{{5}}{{*}}{{{DS_NAME[ds]}}}" if si==0 else ""
            cells=" & ".join(cell(r,m) for m in metrics)
            L.append(f"{pre} & {ST_NAME[strat]} & {cells} \\\\")
        if di<len(DS_ORDER)-1: L.append(r"\midrule")
    L+= [r"\bottomrule",r"\end{tabular}"]
    if note: L.append(f"\\\\[2pt]\n{{\\footnotesize {note}}}")
    L.append(r"\end{table}")
    open(f"{OUT}/{fname}","w").write("\n".join(L)+"\n")
    return fname

# Seed count + dataset coverage are derived from the data so the caption can never
# drift from reality (CIFAR clause only appears when CIFAR is actually present).
_rows=list(csv.DictReader(open(f"{FIN}/aggregated_mean_std.csv")))
_ns=[int(r["n_seeds"]) for r in _rows if r["dataset"] in ("mnist","adult") and r["n_seeds"]]
_N=max(_ns) if _ns else 0
_RANGE=f"{42}--{42+_N-1}"
_has_cifar=any(r["dataset"]=="cifar10" and r.get("n_seeds") for r in _rows)
NOTE_STD=f"MNIST and Adult: mean $\\pm$ std over {_N} seeds ({_RANGE})."
if _has_cifar:
    NOTE_STD+=" CIFAR-10: single seed (42), no std."
NOTE_NA="``---'' = metric not exposed by the strategy."

table("tab_tradeoff.tex",
  "Utility-cost trade-off on the permanent scenarios (S1, S2, S3, S8). "+NOTE_STD,
  "tab:tradeoff",
  ["utility_accuracy_test","js_divergence_mean","activation_cosine_similarity","retrain_fraction"],
  ["Utility","JS div.","Act.\\ cos.","Retrain frac."],
  note=NOTE_NA+" Wall-clock time omitted (hardware-dependent across re-runs).")

table("tab_forgetting.tex",
  "Forgetting-quality metrics, permanent scenarios (S1, S2, S3, S8). "+NOTE_STD,
  "tab:forgetting",
  ["unlearning_score","forget_confidence_mean_unlearned","forget_entropy_mean_unlearned"],
  ["Unl.\\ score","Confidence","Entropy"])

table("tab_mia.tex",
  "Membership inference attack, permanent scenarios (S1, S2, S3, S8). MIA $\\approx 0.5$ is ideal. "+NOTE_STD,
  "tab:mia",
  ["mia_accuracy_unlearned","mia_improvement"],
  ["MIA acc.","MIA impr."])

table("tab_s4.tex",
  "Scenario S4 (temporary exclusion, evaluated at round~3, reported separately from the "
  "permanent scenarios). "+NOTE_STD,
  "tab:s4",
  ["utility_accuracy_test","unlearning_score","mia_accuracy_unlearned","js_divergence_mean"],
  ["Utility","Unl.\\ score","MIA acc.","JS div."],
  source=s4)

# baseline table
br={}
for r in csv.DictReader(open(f"{FIN}/baseline_quality.csv")):
    br.setdefault(r["dataset"],{})[r["metric"]]=(float(r["mean"]),float(r["std"]),int(r["n_seeds"]))
L=[r"\begin{table}[H]",r"\centering",
   "\\caption{Global pre-unlearning model quality (held-out test set), permanent scenarios. "+NOTE_STD+"}",
   r"\label{tab:baseline}",r"\small",r"\begin{tabular}{@{} l c c c c @{}}",r"\toprule",
   r"\textbf{Dataset} & \textbf{Accuracy} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} \\",r"\midrule"]
for ds in DS_ORDER:
    if ds not in br: continue
    b=br[ds]
    def bc(k):
        m,s,n=b[k]; return f"${m:.3f} \\pm {s:.3f}$" if n>1 else f"${m:.3f}$"
    L.append(f"{DS_NAME[ds]} & {bc('accuracy')} & {bc('f1')} & {bc('precision')} & {bc('recall')} \\\\")
L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(f"{OUT}/tab_baseline.tex","w").write("\n".join(L)+"\n")

print("Gegenereerd in", OUT+"/:")
for f in sorted(os.listdir(OUT)): print("  ", f)
