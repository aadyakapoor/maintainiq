import pickle
import numpy as np
from maintainiq.core import ROOT, simulate
from maintainiq.ablation import CoxPH, integrate_rates, allocate_causes, predict_family, cost_grid, paired_interval

def test_cox_breslow_ties_with_known_baseline():
    model=CoxPH().fit(np.zeros((4,3)),np.array([1,1,2,2]),np.array([1,1,0,0]))
    assert np.allclose(model.coef_,0)
    assert np.allclose(model.baseline_,[.5])
    assert np.allclose(model.increments([[0,0,0]])[0,0],.5)

def test_cox_partial_likelihood_stationarity():
    z,t,e,*_=simulate(250,seed=18)
    m=CoxPH().fit(z,t,e>0)
    x=m.scaler.transform(z)
    grad=m.ridge*m.coef_
    for hour in np.unique(t[e>0]):
        risk=x[t>=hour];dead=x[(t==hour)&(e>0)]
        weights=np.exp(risk@m.coef_)
        grad += len(dead)*(weights@risk)/weights.sum()-dead.sum(axis=0)
    assert np.linalg.norm(grad)<1e-3

def test_rate_integration_handles_zero_and_known_rates():
    rates=np.zeros((2,24,5));rates[1,:,0]=.2;rates[1,:,1]=.3
    s,f=integrate_rates(rates)
    assert np.allclose(s[0],1) and np.allclose(f[0],0)
    assert np.isclose(s[1,2],np.exp(-1))
    assert np.isclose(f[1,2,0],.4*(1-np.exp(-1)))
    assert np.allclose(s+f.sum(axis=2),1)
    assert (np.diff(f,axis=1)>=0).all()

def test_saved_family_and_equal_cost_control():
    with (ROOT/"artifacts/ablation_models.pkl").open("rb") as stream:
        family=pickle.load(stream)
    predictions=predict_family(family,[[0,0,0],[1,-1,1]])
    for s,f in predictions.values():
        assert np.allclose(s+f.sum(axis=2),1)
        assert (np.diff(s,axis=1)<=1e-10).all()
        assert (np.diff(f,axis=1)>=-1e-10).all()
    s,f=predictions["Competing-risk discrete"]
    ss,ff=predictions["Same survival + pooled causes"]
    assert np.array_equal(s,ss)
    assert np.allclose(cost_grid(s,f,corrective=[2000]*5),
                       cost_grid(ss,ff,corrective=[2000]*5))

def test_paired_ci_zero_difference():
    assert paired_interval(np.zeros(100))==[0.,0.]

def test_comparison_page_controls():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
    assert not app.exception
    app.radio[0].set_value("Survival comparison").run()
    assert not app.exception
    app.selectbox[0].set_value("Equal cause costs").run()
    app.slider[0].set_value(1.0).run()
    assert not app.exception
    app.selectbox[1].set_value("RNF").run()
    assert not app.exception
