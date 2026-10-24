import numpy as np
from adaptive_ofdm.routing import encode_joint_outcomes,marginal_error_risks,select_branch,pre_neural_features
from adaptive_ofdm.phy import Numerology,transmit
from adaptive_ofdm.classical import pooled_pilot_llr


def test_joint_outcome_truth_table():
    patterns=(np.arange(16)[:,None]>>np.arange(4))&1
    np.testing.assert_array_equal(encode_joint_outcomes(patterns),np.arange(16))
    np.testing.assert_array_equal(marginal_error_risks(np.eye(16)),patterns)


def test_pre_neural_features_independent_of_truth_and_nominal_noise():
    rng=np.random.default_rng(71);cfg=Numerology(symbols=4)
    bits=rng.integers(0,2,(3,cfg.coded_bits))
    obs,truth=transmit(bits,cfg,rng,12)
    classical=pooled_pilot_llr(obs.grid,cfg)
    x=pre_neural_features(obs,classical,cfg)
    truth.coded_bits[:]=0;truth.taps[:]=0;obs.nominal_noise_variance[:]=1e9
    np.testing.assert_array_equal(x,pre_neural_features(obs,classical,cfg))
    assert np.all(np.isfinite(x)) and x.shape==(3,8)


def test_branch_selection_matches_enumerated_costs():
    risks=np.array([[.5,.3,.1,.05],[.3,.2,.1,.01]])
    cost=np.array([1.,1.1,1.3,1.6])
    np.testing.assert_array_equal(select_branch(risks,cost,0),[0,0])
    np.testing.assert_array_equal(select_branch(risks,cost,4),[2,3])


def test_direct_predictor_compaction_preserves_probabilities():
    from adaptive_ofdm.routing import PredictiveRouter
    rng=np.random.default_rng(104)
    models=[{"mean":[.2,.3],"scale":[1.,2.],"weights":rng.standard_normal((3,2)).tolist()} for _ in range(7)]
    router=PredictiveRouter({"columns":[0,1],"models":models},np.arange(7)+1,2.)
    x=rng.standard_normal((20,2))
    expected=np.stack([router.probabilities(m,x)[:,1] for m in router.models],axis=-1)
    np.testing.assert_allclose(router.predict_risks(x),expected,atol=1e-14)
    assert router(x).shape==(20,)
