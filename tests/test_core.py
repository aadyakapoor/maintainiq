import pickle
import numpy as np
import pandas as pd
import pytest
from maintainiq.core import *

def test_features_exclude_targets_and_ids():
    d=pd.read_csv(ROOT/"data"/"ai4i2020.csv").head(5)
    x=features(d)
    d[CODES+["Machine failure"]]=0
    d["UDI"]=-999
    pd.testing.assert_frame_equal(x,features(d))
    assert not set(CODES+["Machine failure","UDI","Product ID"]) & set(x.columns)

@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-1])
def test_invalid_sensor(bad):
    d=pd.read_csv(ROOT/"data"/"ai4i2020.csv").head(1)
    d["Torque [Nm]"]=bad
    with pytest.raises(ValueError):
        features(d)

def test_unknown_quality():
    d=pd.read_csv(ROOT/"data"/"ai4i2020.csv").head(1)
    d["Type"]="X"
    with pytest.raises(ValueError):
        features(d)

def test_competing_probability_conservation():
    b=ConstantHazard(np.array([0]*80+[1]*5+[2]*5+[3]*5+[4]*3+[5]*2))
    s,f=curves(b,[[0,0,0]])
    assert np.allclose(s[0]+f[0].sum(axis=1),1)
    assert np.all(np.diff(s[0])<=0)
    assert np.all(np.diff(f[0],axis=0)>=0)
    assert np.isclose(s[0,4],b.prob[0]**4)
    assert np.allclose(f[0,1],b.prob[1:])

def test_no_failure_cost_prefers_longest_interval():
    p=maintenance_cost(np.ones(25),np.zeros((25,5)),preventive=240)
    assert np.isclose(p.iloc[-1]["Cost per operating hour"],10)
    assert int(p.loc[p["Cost per operating hour"].idxmin(),"Maintenance hour"])==24

def test_certain_first_hour_failure_cost():
    s=np.array([1.,0.,0.]);f=np.zeros((3,5));f[1:,0]=1
    p=maintenance_cost(s,f,300,[100,200,300,400,500])
    assert np.allclose(p["Cost per operating hour"],100)

def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        maintenance_cost(np.ones(25),np.zeros((25,5)),-1)

def test_censoring_and_person_period_alignment():
    z,t,e,x,y,ids=simulate(200,seed=7)
    assert len(x)==len(y)==len(ids)==sum(t)
    assert np.all((t>=1)&(t<=24))
    for i in range(200):
        yi=y[ids==i]
        assert (yi[:-1]==0).all()
        assert yi[-1]==e[i]

def test_reproducible_simulation():
    a=simulate(40,seed=18);b=simulate(40,seed=18)
    for x,y in zip(a,b):
        assert np.array_equal(x,y)

def test_artifact_predicts_and_shap_adds_up():
    with (ROOT/"artifacts"/"models.pkl").open("rb") as f:
        bundle=pickle.load(f)
    row=pd.read_csv(ROOT/"data"/"ai4i2020.csv").iloc[[9000]]
    for code in CODES:
        p=bundle["diagnostics"]["models"][code].predict_proba(features(row))[0,1]
        assert 0<=p<=1
        values,base=explain(bundle["diagnostics"],row,code)
        assert np.isclose(base+values.Contribution.sum(),p,atol=1e-6)
    s,c=curves(bundle["survival"],[[0,0,0]])
    assert np.allclose(s+c.sum(axis=2),1)

def test_streamlit_app_and_interaction():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
    assert not app.exception
    app.radio[0].set_value("Timing & maintenance lab").run()
    app.slider[0].set_value(1.0).run()
    assert not app.exception
    app.radio[0].set_value("Failure diagnosis").run()
    app.button[0].click().run()
    assert not app.exception
