"""Matched single-event versus cause-preserving survival experiments."""
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from .core import ROOT, CODES, HORIZON, simulate, curves, oracle_hazards

class CoxPH:
    """Ridge Cox partial likelihood with Breslow ties, fitted on machine records.
    Uses baseline covariates only. Other causes are censored for cause-specific fits.
    """
    def __init__(self, ridge=0.01):
        self.ridge = ridge

    def fit(self, z, time, event):
        z, time, event = np.asarray(z, float), np.asarray(time), np.asarray(event, bool)
        if len(z) != len(time) or len(time) != len(event) or not event.any():
            raise ValueError("Aligned observations and at least one event are required.")
        self.scaler = StandardScaler().fit(z)
        x = self.scaler.transform(z)
        self.times = np.unique(time[event])
        self.groups = [(x[time >= t], x[(time == t) & event]) for t in self.times]
        def objective(beta):
            loss = .5 * self.ridge * (beta @ beta)
            grad = self.ridge * beta
            for risk, deaths in self.groups:
                eta = risk @ beta
                shift = eta.max()
                weights = np.exp(eta - shift)
                denom = weights.sum()
                count = len(deaths)
                loss += count * (shift + np.log(denom)) - deaths.sum(axis=0) @ beta
                grad += count * (weights @ risk / denom) - deaths.sum(axis=0)
            return loss, grad
        self.objective_ = objective
        fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7})
        if not fit.success:
            raise RuntimeError("Cox fit did not converge: " + fit.message)
        self.coef_ = fit.x
        self.baseline_ = np.array([len(d) / np.exp(r @ fit.x).sum() for r, d in self.groups])
        # Do not retain training rows or an unpicklable closure in saved models.
        del self.objective_, self.groups
        return self

    def increments(self, z):
        x = self.scaler.transform(np.atleast_2d(z))
        increments = np.zeros((len(x), HORIZON))
        for t, base in zip(self.times, self.baseline_):
            increments[:, int(t)-1] = base * np.exp(x @ self.coef_)
        return increments

def integrate_rates(rates):
    """Nonnegative integrated cause rates -> mutually exclusive interval risks.
    Piecewise-constant rate approximation within each one-hour interval.
    """
    rates = np.asarray(rates, float)
    if rates.ndim != 3 or rates.shape[2] != 5 or (rates < 0).any() or not np.isfinite(rates).all():
        raise ValueError("Expected finite, nonnegative [machines, hours, five causes] rates.")
    n, hours, _ = rates.shape
    s, f = np.ones((n, hours+1)), np.zeros((n, hours+1, 5))
    for j in range(hours):
        total = rates[:, j].sum(axis=1)
        share = np.divide(rates[:, j], total[:, None], out=np.zeros((n, 5)), where=total[:, None] > 0)
        mass = -np.expm1(-total)
        f[:, j+1] = f[:, j] + s[:, j, None] * mass[:, None] * share
        s[:, j+1] = s[:, j] * np.exp(-total)
    return s, f

def allocate_causes(s, mix):
    """Decision fallback for an aggregate model; not learned individual CIFs."""
    mix = np.asarray(mix, float)
    if mix.shape != (5,) or (mix < 0).any() or not np.isclose(mix.sum(), 1):
        raise ValueError("Five nonnegative cause shares summing to one are required.")
    return (1 - s[:, :, None]) * mix

def predict_family(bundle, z):
    s, f = curves(bundle["competing_discrete"], z)
    # The binary model classes are 0 (survive) and 1 (any first failure).
    bs, _ = curves(bundle["single_discrete"], z)
    single_rate = bundle["single_cox"].increments(z)
    cs = np.column_stack([np.ones(len(np.atleast_2d(z))), np.exp(-np.cumsum(single_rate, axis=1))])
    rates = np.stack([m.increments(z) for m in bundle["cause_cox"]], axis=2)
    crs, crf = integrate_rates(rates)
    return {
        "Single-event discrete": (bs, allocate_causes(bs, bundle["cause_mix"])),
        "Competing-risk discrete": (s, f),
        "Single-event Cox": (cs, allocate_causes(cs, bundle["cause_mix"])),
        "Cause-specific Cox": (crs, crf),
        "Same survival + pooled causes": (s.copy(), allocate_causes(s, bundle["cause_mix"])),
    }

def fit_family(seed, n=3000):
    z, time, event, x, y, _ = simulate(n, seed=seed)
    competing = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, C=1.0)).fit(x, y)
    single = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, C=1.0)).fit(x, y > 0)
    mix = np.bincount(event[event > 0], minlength=6)[1:].astype(float)
    mix /= mix.sum()
    return {
        "competing_discrete": competing, "single_discrete": single,
        "single_cox": CoxPH().fit(z, time, event > 0),
        "cause_cox": [CoxPH().fit(z, time, event == k) for k in range(1, 6)],
        "cause_mix": mix,
    }

def cost_grid(s, f, preventive=300, corrective=None):
    corrective = np.asarray([1800,2600,2200,3000,1600] if corrective is None else corrective, float)
    if corrective.shape != (5,) or not np.isfinite(corrective).all() or (corrective < 0).any() or not np.isfinite(preventive) or preventive < 0:
        raise ValueError("Costs must be finite and nonnegative, with five corrective costs.")
    return (f[:, 1:] @ corrective + preventive * s[:, 1:]) / np.cumsum(s[:, :-1], axis=1)

def known_curves(z):
    s, f = np.ones((len(z), 25)), np.zeros((len(z), 25, 5))
    for t in range(1, 25):
        p = oracle_hazards(z, t)
        f[:, t] = f[:, t-1] + s[:, t-1, None] * p[:, 1:]
        s[:, t] = s[:, t-1] * p[:, 0]
    return s, f

def paired_interval(values, seed=19):
    """Bootstrap paired test profiles, conditional on one fitted model and simulator."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(500, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]

def run_ablation(seeds=(42, 43, 44), train_n=3000, test_n=1000):
    metric_rows, policy_rows, paired_rows = [], [], []
    first = None
    scenarios = {
        "Default costs": [1800,2600,2200,3000,1600],
        "Equal cause costs": [2200]*5,
        "High heat-failure cost": [1800,8000,2200,3000,1600],
    }
    for seed in seeds:
        print("Survival ablation seed", seed, flush=True)
        bundle = fit_family(seed, train_n)
        if first is None:
            first = bundle
        z, time, event, _, _, _ = simulate(test_n, seed=seed+10000, censor=False)
        predictions = predict_family(bundle, z)
        times = np.arange(25)[None, :]
        truth_s = (time[:, None] > times) | (event[:, None] == 0)
        truth_f = (time[:, None, None] <= times[:, :, None]) & (event[:, None, None] == np.arange(1,6)[None,None,:])
        for name, (s, f) in predictions.items():
            row = {"seed": seed, "model": name,
                   "Survival IBS": float(np.mean((s[:,1:] - truth_s[:,1:])**2)),
                   "Mean cause IBS": float(np.mean((f[:,1:] - truth_f[:,1:])**2)),
                   **{c+" Brier at 24h": float(np.mean((f[:,-1,k] - truth_f[:,-1,k])**2)) for k,c in enumerate(CODES)}}
            metric_rows.append(row)
        os, of = known_curves(z)
        for scenario, costs in scenarios.items():
            actual = cost_grid(os, of, corrective=costs)
            oracle_cost = actual.min(axis=1)
            realized = {}
            for name, (s, f) in predictions.items():
                selected = cost_grid(s, f, corrective=costs).argmin(axis=1)
                values = actual[np.arange(test_n), selected]
                realized[name] = values
                policy_rows.append({"seed":seed, "scenario":scenario, "policy":name,
                                    "Mean cost per hour":float(values.mean()),
                                    "Mean regret vs oracle":float((values-oracle_cost).mean()),
                                    "Mean selected hour":float((selected+1).mean())})
            for label, idx in [("Fixed 12h",11),("Fixed 24h",23)]:
                v=actual[:,idx]
                policy_rows.append({"seed":seed,"scenario":scenario,"policy":label,
                                    "Mean cost per hour":float(v.mean()),
                                    "Mean regret vs oracle":float((v-oracle_cost).mean()),
                                    "Mean selected hour":float(idx+1)})
            for pooled, cause in [("Single-event discrete","Competing-risk discrete"),
                                   ("Single-event Cox","Cause-specific Cox"),
                                   ("Same survival + pooled causes","Competing-risk discrete")]:
                delta = realized[pooled]-realized[cause]
                lo, hi = paired_interval(delta, seed)
                paired_rows.append({"seed":seed,"scenario":scenario,"comparison":pooled+" minus "+cause,
                                    "Mean saving per hour":float(delta.mean()),
                                    "Paired 95% CI lower":lo,"Paired 95% CI upper":hi})
    report = {
        "source":"SIMULATION ONLY; no AI4I event-time or censoring targets are fabricated.",
        "protocol": f"Seeds {list(seeds)}; {train_n} training and {test_n} independent test machines per seed. Identical records within each comparison; no test tuning.",
        "notes":[
            "Discrete comparison uses identical person-period features and regularization, with binary versus six-category outcomes.",
            "Cox comparison uses identical baseline covariates and ridge strength. Breslow ties; competing events are censored in each cause-specific fit.",
            "Cox CIF integration uses piecewise-constant cause-rate allocation per hour. The simulator is discrete and not proportional hazards, so Cox is a deliberately different model family.",
            "Aggregate models use observed training-event cause proportions for cost decisions; these allocated incidences are not learned cause-specific predictions. Censoring can affect those proportions.",
            "The same-survival control keeps aggregate risk identical, isolating the value of cause allocation for unequal costs.",
            "Each paired confidence interval bootstraps test profiles within one seed (500 replicates); it is conditional on the fitted model and simulator, not a universal confidence claim.",
            "Costs are oracle expected per-profile renewal rates, not observed industrial savings. Each row includes all test machines.",
            "RSF is not implemented. The requested direct survival comparison is implemented with Cox and discrete-hazard models.",
        ],
        "metrics":metric_rows, "policies":policy_rows, "paired_comparisons":paired_rows,
    }
    dest=ROOT/"artifacts"
    dest.mkdir(exist_ok=True)
    (dest/"ablation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (dest/"ablation_models.pkl").open("wb") as stream:
        pickle.dump(first,stream)
    for key in ["metrics","policies","paired_comparisons"]:
        pd.DataFrame(report[key]).to_csv(dest/("ablation_"+key+".csv"),index=False)
    summary=pd.DataFrame(metric_rows).groupby("model").mean(numeric_only=True).drop(columns="seed")
    policy=pd.DataFrame(policy_rows).groupby(["scenario","policy"]).mean(numeric_only=True).drop(columns="seed")
    lines=["# Single-event versus competing-risk ablation", "", report["source"], "", report["protocol"], "",
           "Lower Brier scores, expected costs and regrets are better. No improvement is assumed in advance.", "",
           "## Mean prediction metrics across seeds", "", summary.to_csv(), "",
           "## Mean policy outcomes across seeds", "", policy.to_csv(), "", "## Interpretation"]
    lines.extend("- "+v for v in report["notes"])
    (dest/"ABLATION_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    return report

if __name__ == "__main__":
    # Import the canonical module so saved classes never refer to __main__.
    from maintainiq.ablation import run_ablation as execute
    execute()
