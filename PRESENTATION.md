# Demo and viva guide
## A five-minute demonstration
1. Explain the two experiments: AI4I failure diagnosis and independently simulated lifetime prediction.
2. Open Failure diagnosis. Change torque, speed and tool wear. Explain that failure types may coexist and probabilities need not sum to one.
3. Select a failure type and calculate SHAP. Explain positive and negative probability contributions, including the reference value.
4. Open Timing & maintenance lab. Change load or cooling and show cumulative incidence plus survival.
5. Increase preventive replacement cost. Show how the least-cost interval can change.
6. Open Results & baselines. Compare average precision with prevalence and discuss RNF honestly.
7. Show the simulated learned-policy versus fixed-policy evaluation. Call these simulated expected costs, not real savings.

## Questions your teacher may ask
**Where is the novelty?**
The contribution is the integrated decision-support experiment, failure-specific calibrated explanations, cost sensitivity and honest comparison with baselines. No claim that random forests, SHAP or competing risks are new algorithms.

**Why not fit survival directly on AI4I?**
It has no reliable event-time/censoring labels. Tool wear is an input age-like measurement, not remaining useful life. Row numbers are not elapsed hours.

**Why not softmax the five AI4I failure labels?**
Some labels coexist. Softmax would discard valid simultaneous flags. The lifetime experiment uses mutually exclusive first failures, where competing risks are appropriate.

**Does SHAP tell you the cause?**
It tells us how the model assigns a prediction. It does not establish physical causality. Correlated engineered features affect credit allocation.

**Why can random failure be hard to predict?**
The benchmark defines it as unrelated to measured process parameters and it is rare. A strong random-failure prediction claim needs scrutiny.

**What does “when” mean here?**
A distribution over simulated first-failure times. The median can lie beyond the 24-hour observation horizon. It is not a timestamp predicted from AI4I.

**Why use cost per hour?**
We compare repeating replacement cycles of unequal lengths. Cost per cycle alone would favour short cycles without accounting for useful operation.

**What remains before industrial use?**
Real longitudinal records, validated failure definitions, calibrated time-dependent risks, actual intervention costs, imperfect repair and downtime modelling, and prospective policy evaluation.

## Suggested final report structure
Abstract; problem and corrected scope; related work; data audit; multi-label methods; competing-risk simulation; cost objective; experimental protocol; actual results; explanation examples; limitations; future work; references.
Do not copy a claim of numerical improvement until you have checked artifacts/RESULTS.md.

## Demonstrate the added ablation
Open Survival comparison. First compare single-event Cox against cause-specific Cox. Then compare the discrete models. Explain that identical machine data are used within each seed. Switch to Equal cause costs: the same-survival pooled-cause control and competing-risk model have identical costs. Switch to High heat-failure cost to test whether identifying the cause changes the best intervention. Show all three seed results and uncertainty intervals, including any negative or inconclusive outcomes. This is a simulator experiment, not direct survival validation on AI4I.
