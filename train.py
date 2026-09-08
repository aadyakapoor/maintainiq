"""Reproduce all models and reports: python train.py."""
import json
import hashlib
import pickle
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
from maintainiq.core import ROOT, train_diagnostics, train_survival, benchmark_random_split

def main():
    data=ROOT/"data"/"ai4i2020.csv"
    data.parent.mkdir(exist_ok=True)
    if not data.exists():
        url="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
        archive=ROOT/"data"/"ai4i.zip"
        print("Downloading UCI AI4I...",flush=True)
        urllib.request.urlretrieve(url,archive)
        with zipfile.ZipFile(archive) as z:
            data.write_bytes(z.read("ai4i2020.csv"))
    print("Training failure-type classifiers...",flush=True)
    bundle,diag=train_diagnostics(pd.read_csv(data))
    diag["sha256"]=hashlib.sha256(data.read_bytes()).hexdigest()
    print("Training simulated competing risks...",flush=True)
    survival,sim=train_survival()
    dest=ROOT/"artifacts"
    dest.mkdir(exist_ok=True)
    with (dest/"models.pkl").open("wb") as f:
        pickle.dump({"diagnostics":bundle,"survival":survival},f)
    (dest/"diagnostics.json").write_text(json.dumps(diag,indent=2),encoding="utf-8")
    (dest/"simulation.json").write_text(json.dumps(sim,indent=2),encoding="utf-8")
    pd.DataFrame(diag["metrics"]).to_csv(dest/"diagnostic_metrics.csv",index=False)
    pd.DataFrame(sim["metrics"]).to_csv(dest/"survival_metrics.csv",index=False)
    supplemental=benchmark_random_split(pd.read_csv(data))
    (dest/"stratified.json").write_text(json.dumps(supplemental,indent=2),encoding="utf-8")
    lines=["# Measured results", "", "AI4I: first 8,000 rows train; last 2,000 held out. Threshold 0.5.", "", "| Label | Test positives | Average precision | Recall | Brier |", "|---|---:|---:|---:|---:|"]
    for r in diag["metrics"]:
        if r["model"]=="Calibrated forest":
            ap=r["PR AUC (average precision)"]
            lines.append(f"| {r['failure']} | {r['positives']} | {ap if ap is not None else 'N/A'} | {r['Recall at 0.5']:.3f} | {r['Brier']:.5f} |")
    lines.extend(["", "## Simulated policy comparison", "", "Known simulator expected cost per operating hour; average of 300 test profiles, default costs. These are not real savings.", ""])
    for k,v in sim["policy_mean_oracle_cost_per_hour"].items():
        lines.append(f"- {k}: {v:.3f}")
    lines.extend(["", "See JSON/CSV files for all metrics and METHODOLOGY.md for limitations."])
    lines.extend(["", "## Supplementary stratified split", "", supplemental["split"], "", "| Label | Positives | Average precision | Recall at 0.5 |", "|---|---:|---:|---:|"])
    for r in supplemental["metrics"]:
        if r["model"]=="Calibrated forest":
            lines.append(f"| {r['failure']} | {r['positives']} | {r['PR AUC (average precision)']:.4f} | {r['Recall at 0.5']:.3f} |")
    lines.extend(["", "Primary HDF and RNF detection cannot be evaluated: no positives in that holdout. Random failure remains difficult. These synthetic benchmark scores do not establish industrial accuracy."])
    (dest/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    from maintainiq.ablation import run_ablation
    run_ablation()
    print(json.dumps({"AI4I":diag,"simulation":sim},indent=2),flush=True)

if __name__=="__main__":
    main()
