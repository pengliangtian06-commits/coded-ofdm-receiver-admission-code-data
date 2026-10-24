import numpy as np
from adaptive_ofdm.phy import Numerology,make_grid,modulate,demodulate,apply_channel
from adaptive_ofdm.classical import delay_domain_lmmse_llr


def test_delay_lmmse_coherent_flat_no_noise_recovery():
    cfg=Numerology(symbols=4);rng=np.random.default_rng(102)
    bits=rng.integers(0,2,(4,cfg.coded_bits));grid=make_grid(bits,cfg)
    gains=np.array([1+.2j,.3+.8j,-.8+.1j,-.2-.9j])[:,None,None]
    for pooled in [False,True]:
        llr=delay_domain_lmmse_llr(grid*gains,cfg,pooled)
        np.testing.assert_array_equal(llr<0,bits)


def test_delay_lmmse_invariant_to_global_receiver_phase():
    cfg=Numerology(symbols=2);rng=np.random.default_rng(103)
    bits=rng.integers(0,2,(2,cfg.coded_bits));grid=make_grid(bits,cfg)
    noise=(rng.standard_normal(grid.shape)+1j*rng.standard_normal(grid.shape))*.2
    grid=grid+noise
    for pooled in [False,True]:
        original=delay_domain_lmmse_llr(grid,cfg,pooled)
        rotated=delay_domain_lmmse_llr(grid*np.exp(1.37j),cfg,pooled)
        np.testing.assert_allclose(original,rotated,atol=1e-10)


def test_scalar_channel_posterior_variance_and_confidence_boundary():
    from adaptive_ofdm.classical import delay_domain_channel
    cfg=Numerology(symbols=4);rng=np.random.default_rng(106)
    grid=make_grid(rng.integers(0,2,(2,cfg.coded_bits)),cfg)
    grid+=.3*(rng.standard_normal(grid.shape)+1j*rng.standard_normal(grid.shape))
    _,nominal,variance=delay_domain_channel(grid,cfg,True,np.array([1.]),True)
    averaged_noise=nominal.mean(-1)/cfg.symbols
    expected=averaged_noise/(len(cfg.pilots)+averaged_noise)
    np.testing.assert_allclose(variance,np.broadcast_to(expected[:,None,None],variance.shape),atol=1e-12)
    coherent=delay_domain_lmmse_llr(grid,cfg,True,np.array([1.]))
    predictive=delay_domain_lmmse_llr(grid,cfg,True,np.array([1.]),True)
    assert np.all(abs(predictive)<=abs(coherent)+1e-10)
