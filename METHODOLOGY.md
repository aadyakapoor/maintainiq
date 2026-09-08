# Explainable failure diagnosis and cost-aware competing-risk maintenance
## Research question
Can failure-specific probabilities and explanations support more useful inspection decisions, and can a competing-risk lifetime model improve maintenance scheduling over fixed intervals in a controlled experiment?

## The correction to the original pitch
AI4I 2020 is a synthetic snapshot benchmark. Its CSV has no observed remaining life, event timestamps, censoring indicators or reliable machine trajectories. Product IDs and row order must not be converted into failure times. Its five binary failure flags can overlap. We therefore preserve all five labels using multi-label classification rather than forcing them into exclusive classes.

Timing is evaluated in a **separate, explicitly simulated lifetime dataset**. These two experiments share failure-type names but no statistically validated sensor-to-lifetime mapping. The application does not pretend that an AI4I snapshot predicts an actual failure hour.

## AI4I experiment
Six whitelisted input columns; one-hot product quality; three physical interaction features (temperature difference, mechanical power, wear multiplied by torque). IDs and all labels are excluded. Each of five labels has its own random forest, with sigmoid probability calibration fitted by three-fold CV inside the first 80% training partition. A sixth binary machine-failure model serves as the original-task comparator. The final 20% of rows is held out and never used for fitting, tuning or explanation background selection.

The contiguous split reduces adjacent-row mixing, but does not establish a real machine-level or chronological evaluation. Internal calibration folds are not grouped by machine because reliable machine IDs are unavailable. Hyperparameters and the 0.5 reporting threshold are fixed in advance. Average precision, ROC AUC, Brier score, precision, recall and F1 are reported per label, alongside a training-prevalence predictor. Zero-positive test metrics that cannot be interpreted are recorded as null. Rare random failures may perform near chance. Aggregate-label inconsistencies are reported, not silently corrected.

Permutation SHAP explains the entire calibrated predictor separately for each mode. The background contains 40 training observations and the random seed is fixed. Explanation additivity is tested. Correlated raw and engineered features can share credit unpredictably; the explanations are model attributions, not causal effects. Random failure attribution is especially uncertain.

## Competing-risk lifetime experiment
Generate 4,000 independent machines for training and 1,500 different machines for testing. A machine has standardized load, cooling effectiveness and quality. Each operating hour has six mutually exclusive outcomes: survive, or the first failure of one of five causes. A softmax simulator specifies the hazards; its exact coefficients are in core.py. These coefficients and the meaning of an hour are assumptions, not estimates from AI4I.

Training machines have independent administrative follow-up limits between 12 and 24 hours. Machines that survive their limit are right-censored; their observed non-event intervals enter training, but no future labels are invented. The test experiment observes every machine through hour 24, allowing ordinary Brier scoring within that horizon without censoring weights. Survival beyond hour 24 is not observed.

A scaled multinomial logistic regression learns discrete conditional hazards from person-period rows, including age, age squared and age–load interaction. Machine sets are separated before expansion. A constant-hazard model is the baseline. Synthetic data are deliberately compatible with the chosen model class; success here tests implementation, not robustness to real-world model misspecification.

For cause k:
- h_k(t) = probability of cause k at hour t conditional on survival through t-1.
- S(t) = S(t-1) × (1 − sum_k h_k(t)), with S(0)=1.
- F_k(t) = F_k(t-1) + S(t-1) × h_k(t), with F_k(0)=0.
- S(t) + sum_k F_k(t) = 1.

The F curves are cumulative incidence, not five independent survival probabilities. If S never falls to 0.5, median failure time is reported as beyond the horizon.

## Cost model
The policy is age-based perfect replacement under a fixed operating profile. Failures occur at interval ends. Replacement is instantaneous; corrective costs may include a monetized downtime penalty, but calendar downtime is not explicitly modelled.

For a replacement interval tau:
- Expected cycle cost = preventive cost × S(tau) + sum_k corrective cost_k × F_k(tau).
- Expected operating time = sum from t=0 to tau-1 of S(t).
- Renewal cost per operating hour = expected cycle cost / expected operating time.

Choose the smallest-cost interval in the integer grid 1–24 hours. This is an optimum under the specified costs, replacement assumptions and horizon, not a universally optimal real maintenance date. The dashboard varies preventive cost to show sensitivity. The evaluation compares learned intervals, fixed 12-hour and fixed 24-hour policies, and the oracle optimum using known simulator probabilities on 300 independent test profiles. It averages per-profile rates; it is not a fleet-level pooled renewal rate or measured industrial savings.

## Novelty you can defend
The project contribution is an integrated, reproducible evaluation of failure-specific diagnosis, calibrated-model explanations, competing-risk lifetime modelling and cost-sensitive scheduling, with explicit dataset-validity checks and baseline comparisons. The component methods are established. Their combination is an application contribution; claiming a globally new algorithm requires a separate literature review. Do not claim all existing maintenance systems are binary.

Suggested title: **Explainable Failure-Type Diagnosis and Cost-Aware Maintenance Planning: An AI4I Benchmark and Competing-Risk Simulation Study.**

Suggested pitch: “Our system identifies likely failure types and explains its predictions. In a separate lifetime simulation, it estimates which failure occurs first and selects the replacement interval with the lowest expected cost per operating hour.”

## What would make timing an empirical claim?
Obtain machine IDs, observation timestamps, operating age, first-event cause, event/censoring time and maintenance history. Define event precedence for simultaneous faults before fitting. Split by machine and time. Evaluate cumulative-incidence calibration, censoring-adjusted Brier scores and policy outcomes under documented costs. Replace the simulator only after agreeing on this event schema.

## Sources
- UCI AI4I 2020 dataset, DOI 10.24432/C5HS5C, CC BY 4.0: https://archive.ics.uci.edu/dataset/601/ai4i
- Matzka (2020), Explainable Artificial Intelligence for Predictive Maintenance Applications, introductory paper linked on the UCI page.
- Scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html
- SHAP permutation explainer: https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html

## Supplementary split audit
The primary final-20% holdout contains zero HDF and RNF positives. Their detection performance is therefore unassessed by that split. A separate seeded stratified 80/20 split per label provides supplementary metrics for all labels. These row-wise splits can mix adjacent measurements and are potentially optimistic; they overlap the original experiment and are not independent confirmation. They do not select models or thresholds. The dashboard continues to use the original contiguous-split models.


## Implemented direct survival ablation (September 2026)
The Survival comparison page now compares single-event Cox with five cause-specific Cox models, and binary with multinomial discrete-time hazards. Every comparison uses identical simulated training/test machines within a seed. Three seeds (42, 43, 44) each have 3,000 training and 1,000 independent test machines. Training is right-censored; test follow-up is complete through 24 simulated hours.

Cox uses standardized baseline load, cooling and quality, ridge penalty 0.01, Breslow ties and a fitted baseline hazard. Each cause-specific model treats other first failures as censored. Cause rates are integrated with an explicitly documented piecewise-constant within-hour approximation that conserves total probability. The simulator is not proportional hazards; this limits conclusions about Cox performance.

An aggregate survival model cannot predict individual causes. For its maintenance decisions, the baseline allocates failures according to cause proportions in observed training events. The same-survival control holds total risk fixed and changes only this allocation. Equal cause costs must give identical cost curves for that control; a regression test verifies it. Unequal costs expose any decision value from preserving causes.

Results include survival and cause-specific Brier scores, oracle-evaluated policy cost and regret, fixed-interval comparators, and paired 95% bootstrap intervals within each fitted seed. Negative savings or intervals crossing zero must not be reported as an improvement. The bootstrap is conditional on the simulator and trained model; it is not industrial validation.

Random survival forests are not implemented. The direct ablation is implemented using Cox and discrete hazards. The prototype still does not claim observed AI4I lifetime prediction. See artifacts/ABLATION_RESULTS.md and the ablation CSV/JSON files for reproducible results.

Cox method reference: https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html
