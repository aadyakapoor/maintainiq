# Measured results

AI4I: first 8,000 rows train; last 2,000 held out. Threshold 0.5.

| Label | Test positives | Average precision | Recall | Brier |
|---|---:|---:|---:|---:|
| TWF | 10 | 0.059547735437222866 | 0.000 | 0.00497 |
| HDF | 0 | N/A | 0.000 | 0.00000 |
| PWF | 10 | 0.9999999999999999 | 1.000 | 0.00000 |
| OSF | 21 | 1.0000000000000002 | 1.000 | 0.00025 |
| RNF | 0 | N/A | 0.000 | 0.00001 |
| Machine failure | 39 | 0.7826438376826598 | 0.462 | 0.00906 |

## Simulated policy comparison

Known simulator expected cost per operating hour; average of 300 test profiles, default costs. These are not real savings.

- Learned schedule: 108.360
- Fixed 12h: 110.699
- Fixed 24h: 132.277
- Oracle optimum: 108.305

See JSON/CSV files for all metrics and METHODOLOGY.md for limitations.

## Interpretation and supplementary benchmark

The primary holdout has no HDF or RNF positives, so it cannot assess their detection performance. Tool-wear recall is zero at the fixed 0.5 threshold. Very strong power and overstrain scores reflect a small synthetic benchmark whose physical rules are captured by the engineered features; they do not establish industrial accuracy.

The separate stratified split mixes rows and can be optimistic. It is not used to select the deployed models.

| Label | Test positives | Average precision | Recall at 0.5 |
|---|---:|---:|---:|
| TWF | 9 | 0.0440 | 0.000 |
| HDF | 23 | 0.9923 | 0.826 |
| PWF | 19 | 1.0000 | 1.000 |
| OSF | 20 | 0.9917 | 0.900 |
| RNF | 4 | 0.0016 | 0.000 |

RNF average precision is below its prevalence baseline. No successful random-failure detection claim is supported.

13 tests passed, including probability conservation, censoring, cost edge cases, all five SHAP explanations and Streamlit interaction checks.
