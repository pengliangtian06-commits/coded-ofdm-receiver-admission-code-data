import numpy as np
from adaptive_ofdm.channels import cir_to_fir,TDLSettings,TDLWaveformChannel
from adaptive_ofdm.phy import apply_channel,Numerology


def test_time_varying_convolution_reduces_to_static_reference():
    cfg=Numerology(symbols=2);rng=np.random.default_rng(90)
    x=rng.standard_normal((3,160))+1j*rng.standard_normal((3,160))
    h=rng.standard_normal((3,5))+1j*rng.standard_normal((3,5))
    varying=np.repeat(h[:,:,None],x.shape[-1],axis=-1)
    np.testing.assert_allclose(apply_channel(x,varying,cfg),apply_channel(x,h,cfg),atol=1e-12)


def test_single_path_zero_delay_is_exact_group_delay():
    a=np.ones((2,1,20),complex);tau=np.zeros((2,1))
    taps,captured=cir_to_fir(a,tau,1.92e6,-4,11)
    expected=np.zeros((2,16,20),complex);expected[:,4]=1
    np.testing.assert_allclose(taps,expected,atol=1e-15)
    np.testing.assert_allclose(captured,1)


def test_fractional_fir_agrees_with_sionna_reference():
    import torch
    from sionna.phy.channel import cir_to_time_channel
    rng=np.random.default_rng(91)
    a=(rng.standard_normal((2,3,20))+1j*rng.standard_normal((2,3,20))).astype(np.complex64)
    tau=rng.uniform(0,1e-6,(2,3)).astype(np.float32)
    ours,_=cir_to_fir(a,tau,1.92e6)
    reference=cir_to_time_channel(1.92e6,torch.from_numpy(a)[:,None,None,None,None],
        torch.from_numpy(tau)[:,None,None],-4,11,normalize=False).numpy()[:,0,0,0,0].transpose(0,2,1)
    np.testing.assert_allclose(ours,reference,atol=2e-6)


def test_tdl_seed_replay_and_static_speed_boundary():
    first=TDLWaveformChannel(TDLSettings(seed=92));second=TDLWaveformChannel(TDLSettings(seed=92))
    a=first(3,64);b=second(3,64)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(a,np.repeat(a[:,:,:1],64,axis=-1),atol=1e-6)
    assert first.last_diagnostics["captured_independent_path_energy_min"]>.95
