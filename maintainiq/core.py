"""Multi-label diagnostics and a separate simulated competing-risk experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
CODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NAMES = ["Tool wear", "Heat dissipation", "Power", "Overstrain", "Random failure"]
RAW = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
ACTIONS = [
    "Inspect tool condition and consider replacement.",
    "Inspect cooling, ventilation and the temperature difference.",
    "Inspect the drive, electrical supply and mechanical load.",
    "Inspect torque, loading and tool condition; review operating limits.",
    "Inspect logs and carry out general diagnostics; sensors may not explain random events.",
]

def features(df):
    """Whitelist inputs: neither IDs nor any failure labels enter the model."""
    missing = set(RAW) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if not df["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must be L, M or H.")
    x = df[RAW[1:]].apply(pd.to_numeric, errors="raise").astype(float).copy()
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Sensor values must be finite and non-missing.")
    if (x[RAW[1:5]] <= 0).any().any() or (x[RAW[5]] < 0).any():
        raise ValueError("Temperatures, speed and torque must be positive; wear cannot be negative.")
    for kind in ["L", "M", "H"]:
        x["Type " + kind] = (df["Type"] == kind).astype(float)
    x["Temperature difference [K]"] = x[RAW[2]] - x[RAW[1]]
    x["Mechanical power [W]"] = x[RAW[3]] * x[RAW[4]] * 2 * np.pi / 60
    x["Wear x torque [min Nm]"] = x[RAW[5]] * x[RAW[4]]
    return x

def binary_metrics(y, p):
    return {
        "positives": int(np.sum(y)), "n": len(y),
        "PR AUC (average precision)": float(average_precision_score(y, p)) if np.sum(y) else None,
        "ROC AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "Brier": float(brier_score_loss(y, p)),
        "Precision at 0.5": float(precision_score(y, p >= .5, zero_division=0)),
        "Recall at 0.5": float(recall_score(y, p >= .5, zero_division=0)),
        "F1 at 0.5": float(f1_score(y, p >= .5, zero_division=0)),
    }

def train_diagnostics(df):
    for c in CODES + ["Machine failure"]:
        if c not in df or not df[c].isin([0, 1]).all():
            raise ValueError("Expected binary label " + c)
    x = features(df)
    # Contiguous holdout avoids random mixing of adjacent generated sensor readings.
    cut = int(.8 * len(df))
    if cut < 100:
        raise ValueError("At least 125 rows are required.")
    train, test = x.iloc[:cut], x.iloc[cut:]
    models, rows = {}, []
    for code in CODES + ["Machine failure"]:
        print("Fitting " + code, flush=True)
        y = df[code].iloc[:cut]
        if y.value_counts().min() < 3 or y.nunique() < 2:
            raise ValueError("Need at least three training examples in both classes for " + code)
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=42)
        # Internal CV calibration sees only the training partition.
        model = CalibratedClassifierCV(rf, method="sigmoid", cv=3, ensemble=False)
        model.fit(train, y)
        models[code] = model
        truth = df[code].iloc[cut:].to_numpy()
        for name, p in [("Calibrated forest", model.predict_proba(test)[:, 1]),
                        ("Prevalence baseline", np.full(len(test), y.mean()))]:
            rows.append({"failure": code, "model": name, **binary_metrics(truth, p)})
    report = {
        "dataset": "UCI AI4I 2020 (synthetic benchmark)",
        "rows": len(df), "train_rows": cut, "test_rows": len(df)-cut,
        "split": "First 80% in supplied row order train; last 20% held out. Not a machine-level or real temporal validation.",
        "multiple_failure_rows": int((df[CODES].sum(axis=1) > 1).sum()),
        "aggregate_label_disagreements": int((df[CODES].any(axis=1).astype(int) != df["Machine failure"]).sum()),
        "counts": {c:int(df[c].sum()) for c in CODES},
        "metrics": rows,
    }
    return {"models": models, "background": train.sample(min(40,len(train)),random_state=42),
            "ranges": {c:[float(df[c].iloc[:cut].min()),float(df[c].iloc[:cut].max())] for c in RAW[1:]}}, report

def explain(bundle, row, code):
    """SHAP of the entire calibrated predictor, not an unexplained proxy."""
    import os
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import shap
    x = features(row)
    cols = list(x.columns)
    def predict(a):
        return bundle["models"][code].predict_proba(pd.DataFrame(a, columns=cols))[:, 1]
    explainer = shap.Explainer(predict, shap.maskers.Independent(bundle["background"]),
                              algorithm="permutation", seed=42)
    e = explainer(x, max_evals=2 * len(cols) * 12 + 1)
    return pd.DataFrame({"Feature":cols, "Contribution":e.values[0]}), float(e.base_values[0])

SIM_FEATURES = ["load", "cooling", "quality"]
HORIZON = 24

def design(z, times):
    z = np.asarray(z, dtype=float)
    t = np.asarray(times, dtype=float) / HORIZON
    return np.column_stack([z, t, t*t, z[:,0]*t])

def oracle_hazards(z, t):
    """Explicit experimental assumptions. One step = one simulated operating hour."""
    load, cooling, quality = z.T
    age = np.full(len(z), t / HORIZON)
    logits = np.column_stack([
        -6.0 + 4.0*age + .5*load - .25*quality,
        -5.4 + 1.0*load - 1.2*cooling + .3*age,
        -5.5 + 1.3*load + .2*age,
        -6.0 + 1.2*load + 2.4*age + .6*load*age - .3*quality,
        np.full(len(z), -6.2)
    ])
    logits = np.column_stack([np.zeros(len(z)), logits])
    exp = np.exp(logits - logits.max(axis=1,keepdims=True))
    return exp / exp.sum(axis=1,keepdims=True)

def simulate(n=4000, seed=42, censor=True):
    rng = np.random.default_rng(seed)
    z = np.column_stack([rng.normal(0,.8,n), rng.normal(0,.8,n), rng.integers(0,3,n)-1])
    limits = rng.integers(12,HORIZON+1,n) if censor else np.full(n,HORIZON)
    duration = limits.copy()
    event = np.zeros(n,dtype=int)
    alive = np.ones(n,dtype=bool)
    blocks, targets, ids = [], [], []
    for t in range(1,HORIZON+1):
        ix = np.flatnonzero(alive & (limits >= t))
        if len(ix) == 0:
            break
        p = oracle_hazards(z[ix],t)
        draw = (rng.random(len(ix))[:,None] > np.cumsum(p,axis=1)).sum(axis=1)
        blocks.append(design(z[ix],np.full(len(ix),t)))
        targets.extend(draw.tolist())
        ids.extend(ix.tolist())
        failed = ix[draw > 0]
        duration[failed] = t
        event[failed] = draw[draw > 0]
        alive[failed] = False
    return z, duration, event, np.vstack(blocks), np.array(targets), np.array(ids)

def curves(model, z, horizon=HORIZON):
    z = np.atleast_2d(z)
    n = len(z)
    s = np.ones((n,horizon+1))
    cif = np.zeros((n,horizon+1,5))
    for t in range(1,horizon+1):
        raw = model.predict_proba(design(z,np.full(n,t)))
        p = np.zeros((n,6))
        p[:,model.classes_.astype(int)] = raw
        s[:,t] = s[:,t-1]*p[:,0]
        cif[:,t] = cif[:,t-1] + s[:,t-1,None]*p[:,1:]
    return s, cif

class ConstantHazard:
    def __init__(self, targets):
        self.classes_ = np.arange(6)
        self.prob = (np.bincount(targets,minlength=6)+1)/(len(targets)+6)
    def predict_proba(self, x):
        return np.tile(self.prob,(len(x),1))

def maintenance_cost(s, cif, preventive=300., corrective=None):
    """Renewal cost/hour. Failures occur at interval ends; replacements are perfect and instantaneous."""
    s, cif = np.asarray(s), np.asarray(cif)
    corrective = np.array([1800,2600,2200,3000,1600] if corrective is None else corrective,dtype=float)
    if preventive < 0 or corrective.shape != (5,) or (corrective < 0).any() or not np.isfinite(corrective).all() or not np.isfinite(preventive):
        raise ValueError("Provide finite nonnegative costs, with five corrective costs.")
    if s.ndim != 1 or cif.shape != (len(s),5) or len(s)<2:
        raise ValueError("Expected survival and five cumulative incidence curves including time zero.")
    if not np.isfinite(s).all() or not np.isfinite(cif).all() or np.any(s<0) or np.any(s>1) or np.any(cif<0) or not np.isclose(s[0],1) or not np.allclose(cif[0],0):
        raise ValueError("Curves must begin at survival 1, incidence 0 and contain finite probabilities.")
    if not np.allclose(s+cif.sum(axis=1),1,atol=1e-6) or np.any(np.diff(s)>1e-8) or np.any(np.diff(cif,axis=0)<-1e-8):
        raise ValueError("Invalid probability curves.")
    expected_cost = cif[1:] @ corrective + preventive*s[1:]
    operating_time = np.cumsum(s[:-1])
    return pd.DataFrame({"Maintenance hour": np.arange(1,len(s)),
                         "Expected cycle cost":expected_cost,
                         "Expected operating hours":operating_time,
                         "Cost per operating hour":expected_cost/operating_time})

def train_survival():
    z, duration, event, x, y, ids = simulate()
    model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,C=1.0))
    model.fit(x,y)
    baseline = ConstantHazard(y)
    # Completely different machines. Full 24-hour observation for uncensored evaluation.
    tz, td, te, _, _, _ = simulate(1500,seed=2026,censor=False)
    rows = []
    for name,m in [("Competing-risk model",model),("Constant-hazard baseline",baseline)]:
        s,f = curves(m,tz)
        time = np.arange(HORIZON+1)
        survival_truth = td[:,None] > time[None,:]
        survival_truth[:,0] = True
        survival_truth[(te==0),:] = True
        rows.append({"model":name,
                     "Integrated survival Brier (hours 1-24)":float(np.mean((s[:,1:]-survival_truth[:,1:])**2)),
                     **{c+" CIF Brier at 24h":float(np.mean((f[:,-1,k]-((te==k+1)&(td<=24)))**2)) for k,c in enumerate(CODES)}})
    # Evaluate chosen schedules against known simulator probabilities, independently of learned predictions.
    def oracle_curves(z):
        ss=np.ones((len(z),25)); ff=np.zeros((len(z),25,5))
        for t in range(1,25):
            p=oracle_hazards(z,t)
            ss[:,t]=ss[:,t-1]*p[:,0]
            ff[:,t]=ff[:,t-1]+ss[:,t-1,None]*p[:,1:]
        return ss,ff
    ps,pf = curves(model,tz[:300])
    os,of = oracle_curves(tz[:300])
    policy = []
    for i in range(300):
        estimated=maintenance_cost(ps[i],pf[i])["Cost per operating hour"].to_numpy()
        actual=maintenance_cost(os[i],of[i])["Cost per operating hour"].to_numpy()
        policy.append([actual[np.argmin(estimated)],actual[11],actual[23],actual.min()])
    means=np.mean(policy,axis=0)
    report={"source":"SIMULATED lifetimes, not AI4I timing labels",
            "train_machines":len(z),"test_machines":len(tz),
            "train_censored":int((event==0).sum()),"train_events":{c:int((event==k+1).sum()) for k,c in enumerate(CODES)},
            "evaluation":"Independent machines; test follow-up complete through 24h; administrative survival beyond 24h unknown.",
            "metrics":rows,
            "policy_mean_oracle_cost_per_hour":dict(zip(["Learned schedule","Fixed 12h","Fixed 24h","Oracle optimum"],means.tolist())),
            "policy_evaluation":"Mean of 300 per-machine renewal cost rates under known simulator, default costs; not observed savings."}
    return model,report


def benchmark_random_split(df):
    """Supplementary, potentially optimistic row-wise benchmark; never used for model selection."""
    from sklearn.model_selection import train_test_split
    x=features(df)
    results=[]
    for code in CODES:
        print("Supplementary split: " + code, flush=True)
        a,b=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=df[code])
        estimator=RandomForestClassifier(n_estimators=150,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=42)
        model=CalibratedClassifierCV(estimator,method="sigmoid",cv=3,ensemble=False)
        model.fit(x.iloc[a],df[code].iloc[a])
        for name,p in [("Calibrated forest",model.predict_proba(x.iloc[b])[:,1]),("Prevalence baseline",np.full(len(b),df[code].iloc[a].mean()))]:
            results.append({"failure":code,"model":name,**binary_metrics(df[code].iloc[b].to_numpy(),p)})
    return {"split":"Separate seeded 80/20 stratified row split per label. Adjacent readings can mix; potentially optimistic. Overlaps the primary experiment, so it is not independent confirmation. Not used for tuning or dashboard models.","metrics":results}
