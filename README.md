# MaintainIQ
A runnable student research project for explainable predictive maintenance.

**Important:** AI4I supports failure-label diagnosis, not validated failure timing. The separate timing and maintenance laboratory uses simulated lifetimes and is labelled accordingly. Read METHODOLOGY.md before presenting.

## Start on this computer
Double-click START_PROJECT.bat. The trained models and isolated runtime are already prepared. Keep the project in its current folder to use that runtime.

## Start on another Windows computer
Python 3.10 or 3.11 is recommended. Open a terminal in this folder:
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
If your Python is 3.11, replace -3.10 with -3.11. Alternatively double-click setup_and_run.bat; it detects the installed Anaconda Python or uses Python 3.10 through the Windows launcher. The first setup needs internet for dependencies. The original dataset is bundled with attribution. Models can be reproduced with train.py.

## Included
- Five calibrated multi-label random forests plus an aggregate binary comparator.
- Feature engineering without target or ID leakage.
- Per-failure SHAP explanations of the calibrated probability.
- Synthetic right-censored machine lifetimes and multinomial competing hazards.
- Survival and cause-specific cumulative incidence curves.
- Expected renewal-cost optimisation with editable costs and sensitivity.
- Contiguous AI4I holdout; independent simulated test machines; baseline metrics.
- Sensor CSV batch prediction and downloadable results.
- Mathematical, data validation, SHAP and application tests.
- Reproducible model artifacts, evaluation JSON/CSV and a simple presentation guide.

## Files
- app.py: interactive Streamlit application.
- train.py: dataset acquisition, training and evaluation.
- maintainiq/core.py: features, estimators, simulation and cost equations.
- artifacts/: generated models and measured results.
- data/ai4i2020.csv: original UCI data.
- data/example_sensors.csv: upload example using original column names.
- METHODOLOGY.md: assumptions, equations, novelty and limitations.
- PRESENTATION.md: demonstration script and viva preparation.
- tests/: validation checks.

## Run checks
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Train before running the artifact and application tests. The SHAP test can take a minute. The saved model pickle is generated locally; do not replace it with an untrusted downloaded pickle.

## Reproduction notes
Seed 42 is used for fitting/training simulation and 2026 for independent simulation testing. Dependency versions are specified in requirements.txt; the tested full environment is in artifacts/environment.txt. Models are trained with scikit-learn 1.5.2. Retrain after changing library versions.

## Dataset attribution
AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5HS5C
Source: https://archive.ics.uci.edu/dataset/601/ai4i
License: Creative Commons Attribution 4.0, https://creativecommons.org/licenses/by/4.0/
The original data are distributed unchanged. The features generated during training are derived transformations.

## New: direct survival comparison
Open **Survival comparison** in the sidebar. Compare single-event Cox, cause-specific Cox, binary discrete hazards, multinomial competing hazards and the same-survival cause-allocation control. All timing results remain simulated.

Reproduce only this experiment with `python -m maintainiq.ablation`. Full `python train.py` now includes it. Results are saved in artifacts/ablation.json, three CSV tables and artifacts/ABLATION_RESULTS.md. All 19 regression tests passed after this addition.
