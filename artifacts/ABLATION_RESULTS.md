# Single-event versus competing-risk ablation

SIMULATION ONLY; no AI4I event-time or censoring targets are fabricated.

Seeds [42, 43, 44]; 3000 training and 1000 independent test machines per seed. Identical records within each comparison; no test tuning.

Lower Brier scores, expected costs and regrets are better. No improvement is assumed in advance.

## Mean prediction metrics across seeds

model,Survival IBS,Mean cause IBS,TWF Brier at 24h,HDF Brier at 24h,PWF Brier at 24h,OSF Brier at 24h,RNF Brier at 24h
Cause-specific Cox,0.1539086901993401,0.05991802724818249,0.22034394622498907,0.09224712109755927,0.07568683123883252,0.1280683413670198,0.02986175356223976
Competing-risk discrete,0.15362446355599801,0.05989738366349631,0.22013981746776726,0.09235830298382347,0.07572144082756477,0.12793022244457727,0.029886773527047956
Same survival + pooled causes,0.15362446355599801,0.06275869096990568,0.23915607025361454,0.10409059030528138,0.07741141772710407,0.1325689461582694,0.029961930058116735
Single-event Cox,0.15444849847103467,0.06276655549063243,0.24115849507539097,0.1035382655285555,0.07730391234775998,0.13243147042759287,0.029957584766691293
Single-event discrete,0.1541589515111311,0.062757302704943,0.23962031358982008,0.10387400182721464,0.07747721143344992,0.13264343136851078,0.02996032355090365


## Mean policy outcomes across seeds

scenario,policy,Mean cost per hour,Mean regret vs oracle,Mean selected hour
Default costs,Cause-specific Cox,100.8986236175678,0.3069691574317584,12.278666666666666
Default costs,Competing-risk discrete,100.6557682962407,0.06411383610464684,12.157333333333334
Default costs,Fixed 12h,102.56864475217424,1.976990292038188,12.0
Default costs,Fixed 24h,124.08234743740495,23.49069297726891,24.0
Default costs,Same survival + pooled causes,100.70238848975691,0.11073402962088576,11.840000000000002
Default costs,Single-event Cox,101.04764337946067,0.4559889193246173,12.246
Default costs,Single-event discrete,100.90104468798812,0.30939022785207954,11.892333333333333
Equal cause costs,Cause-specific Cox,95.52049176152114,0.232377740755195,12.137
Equal cause costs,Competing-risk discrete,95.34954423458207,0.061430213816115,11.981
Equal cause costs,Fixed 12h,96.66577996916565,1.3776659483997136,12.0
Equal cause costs,Fixed 24h,120.04510449521455,24.756990474448614,24.0
Equal cause costs,Same survival + pooled causes,95.34954423458207,0.061430213816115,11.981
Equal cause costs,Single-event Cox,95.69069566570336,0.40258164493742293,12.385
Equal cause costs,Single-event discrete,95.50161972957808,0.21350570881214007,12.046666666666667
High heat-failure cost,Cause-specific Cox,154.78914840969085,0.6009655106750577,12.094666666666667
High heat-failure cost,Competing-risk discrete,154.37604840361954,0.18786550460372564,11.916666666666666
High heat-failure cost,Fixed 12h,156.3335494647062,2.14536656569037,12.0
High heat-failure cost,Fixed 24h,178.8240545592205,24.635871660204696,24.0
High heat-failure cost,Same survival + pooled causes,154.75586914400535,0.5676862449895245,10.286999999999999
High heat-failure cost,Single-event Cox,154.7043851927593,0.5162022937434613,10.713333333333333
High heat-failure cost,Single-event discrete,154.7342321678003,0.5460492687844701,10.366


## Interpretation
- Discrete comparison uses identical person-period features and regularization, with binary versus six-category outcomes.
- Cox comparison uses identical baseline covariates and ridge strength. Breslow ties; competing events are censored in each cause-specific fit.
- Cox CIF integration uses piecewise-constant cause-rate allocation per hour. The simulator is discrete and not proportional hazards, so Cox is a deliberately different model family.
- Aggregate models use observed training-event cause proportions for cost decisions; these allocated incidences are not learned cause-specific predictions. Censoring can affect those proportions.
- The same-survival control keeps aggregate risk identical, isolating the value of cause allocation for unequal costs.
- Each paired confidence interval bootstraps test profiles within one seed (500 replicates); it is conditional on the fitted model and simulator, not a universal confidence claim.
- Costs are oracle expected per-profile renewal rates, not observed industrial savings. Each row includes all test machines.
- RSF is not implemented. The requested direct survival comparison is implemented with Cox and discrete-hazard models.