import numpy as np
import torch
from adaptive_ofdm.phy import Numerology, transmit
from adaptive_ofdm.neural import MultiExitReceiver, features_from_grid, receiver_loss


def test_intermediate_exit_is_prefix_and_shapes_align():
    cfg = Numerology(symbols=4)
    rng = np.random.default_rng(27)
    bits = rng.integers(0,2,(3,cfg.coded_bits))
    obs,_ = transmit(bits,cfg,rng,12)
    x,base = [torch.from_numpy(v) for v in features_from_grid(obs.grid,cfg)]
    torch.manual_seed(27)
    model = MultiExitReceiver(cfg,width=8,blocks=3,exits=(1,3))
    out = model(x,base)
    prefix = model(x,base,max_depth=1)[0]
    assert len(out)==2 and out[0].shape==(3,cfg.coded_bits)
    torch.testing.assert_close(out[0],prefix)
    shallow, state = model.prefix(x,base)
    deep = model.continuation(state,base)
    torch.testing.assert_close(out[0],shallow)
    torch.testing.assert_close(out[-1],deep)
    torch.testing.assert_close(out[-1],model.final_only(x,base))
    loss=receiver_loss(out,torch.tensor(bits,dtype=torch.float32))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.heads["1"].weight.grad.abs().sum()>0
    assert model.heads["3"].weight.grad.abs().sum()>0


def test_preprocessing_does_not_use_truth_or_nominal_noise():
    cfg=Numerology(symbols=4)
    rng=np.random.default_rng(28)
    bits=rng.integers(0,2,(3,cfg.coded_bits))
    obs,truth=transmit(bits,cfg,rng,12)
    first=features_from_grid(obs.grid,cfg)
    truth.coded_bits[:]=1-truth.coded_bits
    truth.taps[:]=0
    obs.nominal_noise_variance[:]=1e8
    second=features_from_grid(obs.grid,cfg)
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a,b)


def test_bce_sign_convention():
    target=torch.tensor([[0.,1.]])
    good=torch.tensor([[10.,-10.]])
    assert receiver_loss([good],target)<receiver_loss([-good],target)


def test_lean_features_match_training_feature_columns():
    from adaptive_ofdm.policy import lean_observable_features,lean_feature_transform,binary_entropy_from_llr
    cfg=Numerology(symbols=4)
    rng=np.random.default_rng(29)
    obs,_=transmit(rng.integers(0,2,(3,cfg.coded_bits)),cfg,rng,12)
    fake_llr=rng.standard_normal((3,cfg.coded_bits))
    p=obs.grid[...,cfg.pilots%cfg.fft_size]
    scale=np.mean(abs(obs.grid[...,cfg.active%cfg.fft_size])**2,axis=(1,2))
    residual=np.mean(abs(p-p.mean(axis=1,keepdims=True))**2,axis=(1,2))/scale
    np.testing.assert_allclose(lean_observable_features(obs,fake_llr,cfg),
        lean_feature_transform(binary_entropy_from_llr(fake_llr),residual))


def test_classical_control_matches_initial_neural_exits():
    from adaptive_ofdm.classical import pooled_pilot_llr
    cfg=Numerology(symbols=4)
    rng=np.random.default_rng(30)
    obs,_=transmit(rng.integers(0,2,(3,cfg.coded_bits)),cfg,rng,12)
    x,base=[torch.from_numpy(v) for v in features_from_grid(obs.grid,cfg)]
    model=MultiExitReceiver(cfg,width=8,blocks=2,exits=(1,2))
    for output in model(x,base):
        np.testing.assert_allclose(output.detach().numpy(),pooled_pilot_llr(obs.grid,cfg))


def test_matched_frontend_starts_at_strong_classical_reference():
    from adaptive_ofdm.neural import receiver_inputs
    from adaptive_ofdm.classical import blanking_lmmse_llr
    cfg=Numerology(symbols=4);rng=np.random.default_rng(105)
    obs,_=transmit(rng.integers(0,2,(3,cfg.coded_bits)),cfg,rng,12,occupancy=.08)
    x,base=[torch.from_numpy(v) for v in receiver_inputs(obs,cfg,"matched_pdp_blanked")]
    prior=np.exp(-np.arange(5)/2);prior/=prior.sum()
    expected=blanking_lmmse_llr(obs.waveform,cfg,pool_symbols=True,tap_power=prior)
    model=MultiExitReceiver(cfg,width=8,blocks=2,exits=(1,2),frontend="matched_pdp_blanked")
    for output in model(x,base):np.testing.assert_allclose(output.detach().numpy(),expected,atol=3e-6,rtol=1e-6)
