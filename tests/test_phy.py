import numpy as np
from adaptive_ofdm.phy import (Numerology, qpsk_map, qpsk_llr, make_grid,
    modulate, demodulate, apply_channel, markov_bursts, rayleigh_taps,
    ls_channel, equalize_llr, transmit)
from adaptive_ofdm.coding import encode, decode


def test_fft_energy_and_roundtrip():
    cfg = Numerology()
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (3, cfg.coded_bits))
    grid = make_grid(bits, cfg)
    waveform = modulate(grid, cfg)
    np.testing.assert_allclose(demodulate(waveform, cfg), grid, atol=1e-12)
    useful = waveform.reshape(3, cfg.symbols, -1)[..., cfg.cp_length:]
    np.testing.assert_allclose((abs(useful)**2).sum(-1), (abs(grid)**2).sum(-1))


def test_linear_convolution_matches_frequency_response_with_cp():
    cfg = Numerology()
    rng = np.random.default_rng(2)
    grid = make_grid(rng.integers(0, 2, (4, cfg.coded_bits)), cfg)
    taps = rayleigh_taps(rng, 4, length=cfg.cp_length+1)
    received = demodulate(apply_channel(modulate(grid, cfg), taps, cfg), cfg)
    expected = np.fft.fft(taps, cfg.fft_size)[:, None, :]*grid
    np.testing.assert_allclose(received, expected, atol=2e-12)


def test_qpsk_llr_sign_and_scale():
    bits = np.array([[0, 0, 0, 1, 1, 0, 1, 1]])
    llr = qpsk_llr(qpsk_map(bits), 0.5)
    np.testing.assert_array_equal(llr < 0, bits)
    np.testing.assert_allclose(abs(llr), 4.)


def test_markov_stationary_occupancy_and_memory():
    rng = np.random.default_rng(3)
    mask = markov_bursts(rng, 128, 4000, 0.15, 8.)
    assert abs(mask.mean()-0.15) < 0.01
    exits = (mask[:, :-1] & ~mask[:, 1:]).sum()/mask[:, :-1].sum()
    assert abs(exits - 1/8) < 0.01


def test_noise_variance_and_deterministic_replay():
    cfg = Numerology()
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, (500, cfg.coded_bits))
    taps = np.ones((500, 1))
    obs, truth = transmit(bits, cfg, np.random.default_rng(5), 8., taps=taps)
    clean = modulate(make_grid(bits, cfg), cfg)
    empirical = np.mean(abs(obs.waveform-clean)**2)
    assert abs(empirical/obs.nominal_noise_variance.mean()-1) < 0.01
    repeated, _ = transmit(bits, cfg, np.random.default_rng(5), 8., taps=taps)
    np.testing.assert_array_equal(obs.waveform, repeated.waveform)
    assert not truth.burst_mask.any()
    assert not hasattr(obs, "coded_bits") and not hasattr(obs, "taps")


def test_coded_no_noise_multipath_recovery_with_known_csi():
    cfg = Numerology()
    rng = np.random.default_rng(6)
    info = rng.integers(0, 2, (3, cfg.coded_bits//2-2))
    bits = encode(info)
    grid = make_grid(bits, cfg)
    taps = rayleigh_taps(rng, 3)
    received = demodulate(apply_channel(modulate(grid, cfg), taps, cfg), cfg)
    h = np.fft.fft(taps, cfg.fft_size)[:, None, cfg.data % cfg.fft_size]
    llr = equalize_llr(received, h, np.full(3, 1e-5), cfg)
    np.testing.assert_array_equal(decode(llr), info)


def test_ls_flat_channel():
    cfg = Numerology()
    rng = np.random.default_rng(7)
    grid = make_grid(rng.integers(0, 2, (2, cfg.coded_bits)), cfg)
    gain = np.array([0.7+0.2j, -0.3+0.8j])[:, None, None]
    np.testing.assert_allclose(ls_channel(grid*gain, cfg),
                              np.broadcast_to(gain, (2, cfg.symbols, len(cfg.data))))


def test_convolutional_reference_against_exhaustive_ml():
    import itertools
    candidates = np.array(list(itertools.product((0, 1), repeat=5)), np.uint8)
    codewords = encode(candidates)
    rng = np.random.default_rng(8)
    llr = rng.standard_normal((30, codewords.shape[1]))
    metric = np.sum(np.logaddexp(0, -(1-2*codewords.astype(float))[None]*llr[:, None]), -1)
    np.testing.assert_array_equal(decode(llr), candidates[metric.argmin(-1)])


def test_invalid_physical_settings_rejected():
    with np.testing.assert_raises(ValueError):
        Numerology(cp_length=64)
    with np.testing.assert_raises(ValueError):
        markov_bursts(np.random.default_rng(0), 1, 10, .9, 1.)


def test_uncoded_qpsk_awgn_matches_analytic_ber():
    from math import erfc, sqrt
    rng = np.random.default_rng(41)
    bits = rng.integers(0, 2, (1, 400000))
    symbols = qpsk_map(bits)
    for ebno_db in (0, 4):
        ebno = 10**(ebno_db/10)
        n0 = 1/(2*ebno)
        y = symbols+(rng.standard_normal(symbols.shape)+1j*rng.standard_normal(symbols.shape))*sqrt(n0/2)
        observed = np.mean((qpsk_llr(y,n0)<0)!=bits)
        expected = .5*erfc(sqrt(ebno))
        standard_error = sqrt(expected*(1-expected)/bits.size)
        assert abs(observed-expected) < 5*standard_error


def test_ebno_noise_variance_is_payload_independent_and_accounts_for_overhead():
    cfg = Numerology(symbols=4)
    bits = np.random.default_rng(42).integers(0, 2, (8, cfg.coded_bits))
    obs,_ = transmit(bits,cfg,np.random.default_rng(43),10,information_bits=cfg.coded_bits//2)
    # Independent numeric reference for the deterministic pilot CP energy.
    expected = 271.74748120364274/(cfg.coded_bits//2)/10
    np.testing.assert_allclose(obs.nominal_noise_variance,expected)


def test_expected_energy_with_fixed_pilots_matches_payload_ensemble():
    from adaptive_ofdm.phy import expected_frame_energy,make_grid,modulate
    cfg=Numerology(symbols=4)
    rng=np.random.default_rng(541)
    bits=rng.integers(0,2,(5000,cfg.coded_bits),dtype=np.uint8)
    waveform=modulate(make_grid(bits,cfg),cfg)
    assert abs(np.mean(np.sum(abs(waveform)**2,axis=-1))/expected_frame_energy(cfg)-1)<.003
