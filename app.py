import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from maintainiq.core import (
    ROOT,
    CODES,
    NAMES,
    RAW,
    ACTIONS,
    features,
    explain,
    curves,
    maintenance_cost,
)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MaintainIQ | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #07111f;
            --panel: #0d1b2a;
            --panel-soft: #102338;
            --border: rgba(148, 163, 184, 0.16);
            --text: #e5edf7;
            --muted: #91a4bb;
            --accent: #25c2b5;
            --accent-2: #67e8f9;
            --danger: #fb7185;
            --warning: #fbbf24;
            --success: #34d399;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 5%, rgba(37, 194, 181, 0.12), transparent 28%),
                radial-gradient(circle at 90% 0%, rgba(56, 189, 248, 0.08), transparent 25%),
                var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.72);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] {
            background: #091522;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            color: #f8fbff !important;
            letter-spacing: -0.02em;
        }

        p, label, .stMarkdown {
            color: var(--text);
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(37, 194, 181, 0.15), rgba(56, 189, 248, 0.05)),
                #0c1a29;
            border: 1px solid rgba(37, 194, 181, 0.22);
            border-radius: 24px;
            padding: 30px 32px;
            margin-bottom: 22px;
            box-shadow: 0 20px 55px rgba(0, 0, 0, 0.24);
        }

        .eyebrow {
            color: #67e8f9;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero-title {
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.045em;
            margin-bottom: 12px;
        }

        .hero-subtitle {
            color: #a8bbd1;
            max-width: 880px;
            font-size: 1rem;
            line-height: 1.65;
        }

        .surface {
            background: rgba(13, 27, 42, 0.88);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .section-kicker {
            color: #67e8f9;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        .section-title {
            color: #f8fbff;
            font-size: 1.45rem;
            font-weight: 750;
            margin-bottom: 3px;
        }

        .section-copy {
            color: #90a4ba;
            font-size: 0.90rem;
            margin-bottom: 12px;
        }

        .mini-card {
            background: linear-gradient(180deg, rgba(16, 35, 56, 0.96), rgba(11, 27, 43, 0.96));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 18px;
            min-height: 115px;
        }

        .mini-label {
            color: #8fa4ba;
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .mini-value {
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 7px;
            line-height: 1.1;
        }

        .mini-foot {
            color: #7f95ad;
            font-size: 0.78rem;
            margin-top: 8px;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .badge-good {
            background: rgba(52, 211, 153, 0.13);
            color: #6ee7b7;
            border: 1px solid rgba(52, 211, 153, 0.28);
        }

        .badge-warn {
            background: rgba(251, 191, 36, 0.12);
            color: #fcd34d;
            border: 1px solid rgba(251, 191, 36, 0.26);
        }

        .badge-critical {
            background: rgba(251, 113, 133, 0.13);
            color: #fda4af;
            border: 1px solid rgba(251, 113, 133, 0.26);
        }

        .sidebar-brand {
            padding: 6px 2px 18px 2px;
        }

        .sidebar-brand-title {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 850;
            letter-spacing: -0.035em;
        }

        .sidebar-brand-copy {
            color: #7890a9;
            font-size: 0.78rem;
            margin-top: 4px;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 35, 56, 0.96), rgba(11, 27, 43, 0.96));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: none;
        }

        div[data-testid="stMetricLabel"] {
            color: #8ea3ba;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
        }

        div[data-testid="stMetricDelta"] {
            color: #67e8f9;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {
            background: #0c1c2c !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
        }

        .stButton > button {
            border-radius: 11px;
            min-height: 42px;
            font-weight: 750;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #18a89d, #0ea5a6);
            color: white;
            border: none;
        }

        .stDownloadButton > button {
            background: linear-gradient(90deg, #18a89d, #0ea5a6) !important;
            color: #ffffff !important;
            border: 1px solid rgba(103, 232, 249, 0.24) !important;
            border-radius: 11px !important;
            min-height: 40px;
            font-weight: 750 !important;
            transition: all 0.18s ease-in-out;
            box-shadow: 0 8px 22px rgba(14, 165, 166, 0.16);
        }

        .stDownloadButton > button p,
        .stDownloadButton > button span {
            color: #ffffff !important;
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #20b8ab, #14b8b8) !important;
            color: #ffffff !important;
            border-color: rgba(103, 232, 249, 0.50) !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 28px rgba(14, 165, 166, 0.24);
        }

        .stDownloadButton > button:hover p,
        .stDownloadButton > button:hover span {
            color: #ffffff !important;
        }

        .stDownloadButton > button:disabled {
            background: #163247 !important;
            color: #8fa4ba !important;
            border-color: rgba(148, 163, 184, 0.14) !important;
            opacity: 1 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }

        .stDownloadButton > button:disabled p,
        .stDownloadButton > button:disabled span {
            color: #8fa4ba !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stFileUploader"] {
            background: rgba(13, 27, 42, 0.72);
            border-radius: 14px;
        }

        .stAlert {
            border-radius: 14px;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.12) !important;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def style_fig(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd8e6"),
        title_font=dict(color="#f8fbff", size=17),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aabbd0"),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.10)",
            zerolinecolor="rgba(148,163,184,0.10)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.08)",
        ),
    )
    return fig


def section_header(kicker, title, copy=""):
    st.markdown(
        f"""
        <div style="margin: 6px 0 14px 0;">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
            <div class="mini-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_state(probability):
    if probability >= 0.65:
        return "Critical", "badge-critical"
    if probability >= 0.35:
        return "Watch", "badge-warn"
    return "Stable", "badge-good"


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
path = ROOT / "artifacts" / "models.pkl"

if not path.exists():
    st.error("Model artifacts are missing.")
    st.info("Run `python train.py` in the project folder, then restart the app.")
    st.stop()


@st.cache_resource
def load_models():
    # Load only this locally generated, trusted artifact; never accept uploaded pickles.
    with path.open("rb") as f:
        return pickle.load(f)


models = load_models()
diag = json.loads((ROOT / "artifacts" / "diagnostics.json").read_text())
sim = json.loads((ROOT / "artifacts" / "simulation.json").read_text())

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">⚙️ MaintainIQ</div>
            <div class="sidebar-brand-copy">
                Explainable predictive maintenance and intervention planning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Workspace",
        [
            "Failure diagnosis",
            "Timing & maintenance lab",
            "Results & baselines",
            "Survival comparison",
            "Project methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("##### System status")
    st.success("Models loaded")
    st.caption("AI4I diagnostic models + simulated lifetime model")

    st.divider()
    st.caption(
        "Research prototype • predictions support inspection decisions and do not establish physical causation."
    )

# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Explainable maintenance intelligence</div>
        <div class="hero-title">Know the failure.<br>Plan the intervention.</div>
        <div class="hero-subtitle">
            Diagnose five failure modes, inspect the drivers behind each prediction,
            and test cost-aware maintenance intervals in a transparent simulation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# PAGE 1: FAILURE DIAGNOSIS
# =====================================================================
if page == "Failure diagnosis":
    section_header(
        "Live assessment",
        "Assess a sensor snapshot",
        "AI4I synthetic benchmark • probabilities represent current failure labels, not future-time risk.",
    )

    left, right = st.columns([0.88, 1.62], gap="large")

    with left:
        st.markdown("#### Machine inputs")
        kind = st.selectbox("Product quality", ["L", "M", "H"])
        air = st.number_input(
            "Air temperature (K)", min_value=250.0, max_value=400.0, value=300.0
        )
        process = st.number_input(
            "Process temperature (K)", min_value=250.0, max_value=500.0, value=310.0
        )
        rpm = st.number_input(
            "Rotational speed (rpm)", min_value=1.0, max_value=10000.0, value=1500.0
        )
        torque = st.number_input(
            "Torque (Nm)", min_value=0.1, max_value=200.0, value=40.0
        )
        wear = st.number_input(
            "Tool wear (min)", min_value=0.0, max_value=1000.0, value=120.0
        )

    row = pd.DataFrame(
        [[kind, air, process, rpm, torque, wear]],
        columns=RAW,
    )
    x = features(row)
    probs = [
        models["diagnostics"]["models"][code].predict_proba(x)[0, 1]
        for code in CODES
    ]

    index = int(np.argmax(probs))
    top_prob = float(probs[index])
    state_name, state_class = risk_state(top_prob)

    with right:
        warnings = []
        for col, (lo, hi) in models["diagnostics"]["ranges"].items():
            if not lo <= float(row[col].iloc[0]) <= hi:
                warnings.append(col)

        if warnings:
            st.warning(
                "Outside training range: "
                + ", ".join(warnings)
                + ". Treat this prediction as extrapolation."
            )

        chart_df = pd.DataFrame(
            {"Failure type": NAMES, "Probability": probs}
        ).sort_values("Probability")

        fig = px.bar(
            chart_df,
            x="Probability",
            y="Failure type",
            orientation="h",
            range_x=[0, 1],
            text=chart_df["Probability"].map(lambda v: f"{v:.1%}"),
        )
        fig.update_traces(
            marker_color="#25c2b5",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(title="Failure probability profile")
        st.plotly_chart(
            style_fig(fig, 395),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            card("Highest probability", NAMES[index], f"{top_prob:.1%}")
        with m2:
            card(
                "Risk state",
                state_name,
                "Relative dashboard interpretation",
            )
        with m3:
            card(
                "Recommended action",
                "Inspect",
                ACTIONS[index],
            )

        st.markdown(
            f'<div style="margin-top:12px;"><span class="badge {state_class}">{state_name}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Failure labels may coexist, so these probabilities do not need to sum to one."
        )

    st.divider()

    section_header(
        "Interpretability",
        "Explain a failure-type prediction",
        "Inspect approximate SHAP contributions for the calibrated predictor.",
    )

    exp_left, exp_right = st.columns([0.55, 1.45], gap="large")

    with exp_left:
        code = st.selectbox(
            "Failure type",
            CODES,
            format_func=lambda v: NAMES[CODES.index(v)],
        )
        run_explanation = st.button(
            "Calculate SHAP explanation",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Contributions explain model behaviour, not physical causation."
        )

    with exp_right:
        if run_explanation:
            with st.spinner("Explaining the calibrated model..."):
                values, base = explain(models["diagnostics"], row, code)

            fig = px.bar(
                values.sort_values("Contribution"),
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Contribution",
                color_continuous_scale="Tealrose",
            )
            fig.update_layout(title="Feature contribution profile")
            st.plotly_chart(
                style_fig(fig, 390),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.info(
                f"Reference probability {base:.4f} + contributions "
                f"{values.Contribution.sum():.4f} = prediction "
                f"{probs[CODES.index(code)]:.4f}"
            )
            st.caption(
                "Approximate permutation SHAP. Engineered and original features are correlated."
            )
        else:
            st.markdown(
                """
                <div class="surface">
                    <div class="mini-label">Explanation panel</div>
                    <div style="font-size:1.15rem;font-weight:750;color:#fff;margin-top:8px;">
                        Choose a failure type and calculate its explanation.
                    </div>
                    <div class="mini-foot">
                        The chart will show which features pushed the model prediction up or down.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    section_header(
        "Batch workflow",
        "Batch diagnosis",
        "Upload up to 10,000 rows using the six original AI4I feature columns.",
    )

    upload = st.file_uploader(
        "Upload sensor CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if upload:
        try:
            batch = pd.read_csv(upload)

            if len(batch) > 10000:
                raise ValueError("Please upload at most 10,000 rows.")

            bx = features(batch)
            result = batch.copy()

            for code in CODES:
                result[code + " probability"] = models["diagnostics"]["models"][
                    code
                ].predict_proba(bx)[:, 1]

            st.dataframe(
                result.head(30),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download predictions",
                result.to_csv(index=False),
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )

        except (ValueError, KeyError, pd.errors.ParserError) as e:
            st.error(str(e))

# =====================================================================
# PAGE 2: TIMING & MAINTENANCE LAB
# =====================================================================
elif page == "Timing & maintenance lab":
    st.warning(
        "SIMULATION ONLY — AI4I contains no observed failure-time targets. "
        "The hours below belong to an independent experimental simulator."
    )

    section_header(
        "Scenario builder",
        "Choose a simulated operating profile",
        "Change operating conditions and inspect the resulting intervention strategy.",
    )

    cols = st.columns(3)
    load = cols[0].slider("Relative load", -2.0, 2.0, 0.0, 0.1)
    cooling = cols[1].slider("Cooling effectiveness", -2.0, 2.0, 0.0, 0.1)
    quality = cols[2].select_slider(
        "Quality factor",
        options=[-1, 0, 1],
        value=0,
    )

    with st.expander(
        "Maintenance costs — editable experimental assumptions",
        expanded=True,
    ):
        preventive = st.number_input(
            "Preventive replacement cost (currency units)",
            min_value=0.0,
            value=300.0,
        )

        costs = []
        for col, name, default in zip(
            st.columns(5),
            NAMES,
            [1800.0, 2600.0, 2200.0, 3000.0, 1600.0],
        ):
            costs.append(
                col.number_input(
                    name + " corrective cost",
                    min_value=0.0,
                    value=default,
                )
            )

    s, f = curves(models["survival"], [[load, cooling, quality]])
    plan = maintenance_cost(s[0], f[0], preventive, costs)
    best = plan.loc[plan["Cost per operating hour"].idxmin()]
    median = np.flatnonzero(s[0] <= 0.5)
    k = int(np.argmax(f[0, -1]))

    section_header(
        "Recommended policy",
        "Cost-aware intervention summary",
        "Optimal only among the tested integer intervals from 1–24 simulated hours.",
    )

    cols = st.columns(4)
    with cols[0]:
        card(
            "Best interval",
            f"{int(best['Maintenance hour'])} h",
            "Lowest-cost tested interval",
        )
    with cols[1]:
        card(
            "Cost / operating hour",
            f"{best['Cost per operating hour']:.2f}",
            "Expected renewal cost",
        )
    with cols[2]:
        card(
            "Median failure time",
            f"{median[0]} h" if len(median) else "> 24 h",
            "Simulated lifetime",
        )
    with cols[3]:
        card(
            "Dominant cause",
            NAMES[k],
            f"{f[0, -1, k]:.1%} cumulative incidence",
        )

    frame = pd.DataFrame(f[0], columns=NAMES)
    frame["Survival"] = s[0]
    frame["Simulated hour"] = np.arange(25)

    l, r = st.columns(2, gap="large")

    with l:
        fig = px.line(
            frame,
            x="Simulated hour",
            y=NAMES + ["Survival"],
            range_y=[0, 1],
            title="First-failure cumulative incidence & survival",
        )
        st.plotly_chart(
            style_fig(fig, 420),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with r:
        fig = px.line(
            plan,
            x="Maintenance hour",
            y="Cost per operating hour",
            markers=True,
            title="Expected renewal cost",
        )
        fig.update_traces(line_color="#25c2b5")
        st.plotly_chart(
            style_fig(fig, 420),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.caption(
        "Replacements reset the same operating profile and are assumed perfect and instantaneous. "
        "Corrective costs may include downtime penalties."
    )

    dl1, dl2 = st.columns([0.55, 1.45])
    with dl1:
        st.download_button(
            "Download maintenance cost curve",
            plan.to_csv(index=False),
            "simulation_maintenance_plan.csv",
            "text/csv",
            use_container_width=True,
        )

    with st.expander("Cost sensitivity"):
        rows = []
        for multiplier in [0.5, 1.0, 2.0]:
            p = maintenance_cost(
                s[0],
                f[0],
                preventive * multiplier,
                costs,
            )
            opt = p.loc[p["Cost per operating hour"].idxmin()]
            rows.append(
                {
                    "Preventive cost multiplier": multiplier,
                    "Best simulated hour": int(opt["Maintenance hour"]),
                    "Cost per hour": opt["Cost per operating hour"],
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

# =====================================================================
# PAGE 3: RESULTS & BASELINES
# =====================================================================
elif page == "Results & baselines":
    section_header(
        "Evaluation",
        "Measured holdout performance",
        "Model quality, rare-event baselines, and simulator evaluation.",
    )

    st.info(diag["split"])

    cols = st.columns(3)
    with cols[0]:
        card("AI4I rows", f"{diag['rows']:,}", "Evaluation dataset")
    with cols[1]:
        card(
            "Multi-failure rows",
            f"{diag['multiple_failure_rows']:,}",
            "Rows carrying multiple failure types",
        )
    with cols[2]:
        card(
            "Label disagreements",
            f"{diag['aggregate_label_disagreements']:,}",
            "Aggregate-label disagreements",
        )

    st.markdown("#### Diagnostic model metrics")
    st.dataframe(
        pd.DataFrame(diag["metrics"]),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "PR AUC is average precision. Compare it with the prevalence baseline; "
        "accuracy alone can hide rare failures. RNF is intentionally difficult and has very few positives."
    )

    secondary = ROOT / "artifacts" / "stratified.json"

    if secondary.exists():
        extra = json.loads(secondary.read_text())
        st.markdown("#### Supplementary stratified benchmark")
        st.warning(extra["split"])
        st.dataframe(
            pd.DataFrame(extra["metrics"]),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    section_header(
        "Simulation",
        "Simulated lifetime results",
        "Performance of the lifetime model and maintenance policy under the experimental simulator.",
    )

    st.write(sim["evaluation"])
    st.dataframe(
        pd.DataFrame(sim["metrics"]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Policy evaluation")
    st.write(sim["policy_evaluation"])
    st.dataframe(
        pd.DataFrame(
            list(sim["policy_mean_oracle_cost_per_hour"].items()),
            columns=["Policy", "Mean cost per hour"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download complete evaluation",
        json.dumps(
            {"diagnostics": diag, "simulation": sim},
            indent=2,
        ),
        "evaluation.json",
        "application/json",
    )

# =====================================================================
# PAGE 4: METHODOLOGY
# =====================================================================

elif page == "Survival comparison":
    from maintainiq.ablation import predict_family, cost_grid
    st.warning("SIMULATION STUDY — these results do not estimate observed AI4I failure times.")
    section_header("Research ablation", "What do we gain by preserving failure types?",
                   "Matched machines, covariates, follow-up and maintenance costs.")
    report_path = ROOT / "artifacts" / "ablation.json"
    if not report_path.exists():
        st.info("Run python -m maintainiq.ablation to prepare the comparison.")
        st.stop()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.write(report["protocol"])
    st.caption("Cox and discrete-time models are implemented. Random survival forests are not included.")
    metrics = pd.DataFrame(report["metrics"])
    mean_metrics = metrics.groupby("model").mean(numeric_only=True).drop(columns="seed").reset_index()
    st.subheader("Prediction accuracy across three simulation seeds")
    st.dataframe(mean_metrics, hide_index=True, use_container_width=True)
    st.caption("Lower Brier scores are better. For aggregate models, cause incidences are allocated using training-event proportions; they are not learned individual cause predictions.")
    st.subheader("Maintenance decisions")
    scenario = st.selectbox("Evaluation cost scenario",
                            ["Default costs", "Equal cause costs", "High heat-failure cost"])
    policies = pd.DataFrame(report["policies"])
    chosen = policies[policies["scenario"] == scenario]
    average = chosen.groupby("policy").mean(numeric_only=True).drop(columns="seed").reset_index()
    fig = px.bar(average, x="Mean cost per hour", y="policy", orientation="h",
                 title="Expected simulator cost — lower is better")
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    st.dataframe(average, hide_index=True, use_container_width=True)
    paired = pd.DataFrame(report["paired_comparisons"])
    st.subheader("Paired comparisons and uncertainty")
    st.dataframe(paired[paired["scenario"] == scenario], hide_index=True, use_container_width=True)
    st.caption("Positive savings favour preserving causes; negative savings favour the aggregate model. An interval crossing zero does not establish an improvement. Intervals resample test profiles within a fitted seed.")
    st.info("The 'Same survival + pooled causes' control keeps total failure risk identical to the competing-risk model. Under equal failure costs, its cost curve must be identical. Under unequal costs, this isolates the value of knowing the cause.")
    st.divider()
    st.subheader("Compare models for one simulated machine")
    inputs = st.columns(3)
    load = inputs[0].slider("Comparison load", -2.0, 2.0, 0.0, .1)
    cooling = inputs[1].slider("Comparison cooling", -2.0, 2.0, 0.0, .1)
    quality = inputs[2].select_slider("Comparison quality", options=[-1,0,1], value=0)
    with (ROOT / "artifacts" / "ablation_models.pkl").open("rb") as stream:
        family = pickle.load(stream)
    predictions = predict_family(family, [[load,cooling,quality]])
    cause = st.selectbox("Cumulative incidence to compare", CODES)
    plot_rows, schedule_rows = [], []
    costs = {"Default costs":[1800,2600,2200,3000,1600],
             "Equal cause costs":[2200]*5,
             "High heat-failure cost":[1800,8000,2200,3000,1600]}[scenario]
    for name, (survival, incidence) in predictions.items():
        for t in range(25):
            plot_rows.append({"Model":name, "Simulated hour":t, "Survival":survival[0,t],
                              "Cause incidence":incidence[0,t,CODES.index(cause)]})
        grid = cost_grid(survival, incidence, corrective=costs)[0]
        schedule_rows.append({"Model":name, "Selected simulated hour":int(grid.argmin()+1),
                              "Model-estimated cost per hour":float(grid.min())})
    l,r=st.columns(2)
    plot_data=pd.DataFrame(plot_rows)
    l.plotly_chart(style_fig(px.line(plot_data,x="Simulated hour",y="Survival",color="Model",
                                    range_y=[0,1])),use_container_width=True)
    r.plotly_chart(style_fig(px.line(plot_data,x="Simulated hour",y="Cause incidence",color="Model",
                                    range_y=[0,1],title=cause+" cumulative incidence")),use_container_width=True)
    st.dataframe(pd.DataFrame(schedule_rows),hide_index=True,use_container_width=True)
    st.caption("Interactive models use seed 42. Costs here are each model's estimates; the evaluation above instead judges chosen schedules using known simulator risks.")
    with st.expander("Protocol and limitations"):
        for note in report["notes"]:
            st.write("• "+note)
    st.download_button("Download survival ablation",json.dumps(report,indent=2),
                       "survival_ablation.json","application/json")

else:
    section_header(
        "Documentation",
        "Project methodology",
        "Research assumptions, model design, evaluation protocol, and limitations.",
    )
    st.markdown((ROOT / "METHODOLOGY.md").read_text(encoding="utf-8"))
