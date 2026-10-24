"""Information-access and selected-path execution checks for frozen study."""
import numpy as np
from adaptive_ofdm.phy import Numerology,transmit,Observation
from adaptive_ofdm.admission import admission_features,llrs


def test_features_do_not_use_simulator_noise_truth():
    cfg=Numerology(symbols=4);rng=np.random.default_rng(504)
    bits=rng.integers(0,2,(3,cfg.coded_bits),dtype=np.uint8)
    obs,_=transmit(bits,cfg,rng,12)
    hidden=Observation(obs.waveform,obs.grid,None)
    np.testing.assert_array_equal(admission_features(obs,cfg),admission_features(hidden,cfg))
    cheap=admission_features(hidden,cfg,True)
    np.testing.assert_array_equal(cheap[:,:2],admission_features(hidden,cfg)[:,:2])
    assert np.isfinite(cheap).all()


def test_fixed_conventional_path_never_touches_neural_objects():
    cfg=Numerology(symbols=4);rng=np.random.default_rng(505)
    obs,_=transmit(rng.integers(0,2,(2,cfg.coded_bits)),cfg,rng,12)
    class Forbidden:
        def __getattribute__(self,name):raise AssertionError('NN executed on a conventional-only path')
    for index in range(7):
        result=llrs(obs,cfg,Forbidden(),Forbidden(),only=index)
        assert list(result)==[index]
        assert result[index].shape==(2,cfg.coded_bits)
        assert np.isfinite(result[index]).all()
