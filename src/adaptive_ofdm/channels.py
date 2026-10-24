"""3GPP TDL coefficients with an explicitly finite fractional-delay frontend.

The path statistics are Sionna TR 38.901 TDL. The discrete waveform uses a
finite sinc FIR with a documented group delay, not an infinite ideal filter.
"""
from dataclasses import asdict,dataclass
import numpy as np


def cir_to_fir(coefficients,delays,sample_rate,l_min=-4,l_max=11):
    """Convert [B,path,time] coefficients and [B,path] delays to [B,lag,time]."""
    a=np.asarray(coefficients);tau=np.asarray(delays)
    if a.ndim!=3 or tau.shape!=a.shape[:2] or sample_rate<=0 or l_min>l_max:
        raise ValueError("invalid CIR/FIR shapes or parameters")
    lags=np.arange(l_min,l_max+1)
    weights=np.sinc(lags[None,None,:]-tau[:,:,None]*sample_rate)
    taps=np.einsum('bpt,bpl->blt',a,weights,optimize=True)
    # Expected independent-path energy fraction of the finite sinc window.
    path_power=np.mean(abs(a)**2,axis=-1)
    captured=(path_power*np.sum(weights**2,axis=-1)).sum(-1)/np.maximum(path_power.sum(-1),1e-12)
    return taps,captured


@dataclass(frozen=True)
class TDLSettings:
    profile:str="A"
    delay_spread_s:float=100e-9
    carrier_frequency_hz:float=3.5e9
    sample_rate_hz:float=1.92e6
    speed_m_per_s:float=0.
    l_min:int=-4
    l_max:int=11
    seed:int=701


class TDLWaveformChannel:
    def __init__(self,settings=TDLSettings()):
        import torch
        from sionna.phy.channel.tr38901 import TDL
        self.settings=settings
        self.model=TDL(model=settings.profile,delay_spread=settings.delay_spread_s,
            carrier_frequency=settings.carrier_frequency_hz,min_speed=settings.speed_m_per_s,
            max_speed=settings.speed_m_per_s,device="cpu",precision="single")
        self.state=torch.Generator(device="cpu").manual_seed(settings.seed).get_state()
        self.last_diagnostics=None

    def __call__(self,batch,samples):
        s=self.settings
        # Sionna exposes a shared RNG: isolate and restore it explicitly.
        generator=self.model.torch_rng
        previous=generator.get_state()
        try:
            generator.set_state(self.state)
            a,tau=self.model(batch,samples,s.sample_rate_hz)
            self.state=generator.get_state()
        finally:
            generator.set_state(previous)
        a=a[:,0,0,0,0].numpy();tau=tau[:,0,0].numpy()
        taps,captured=cir_to_fir(a,tau,s.sample_rate_hz,s.l_min,s.l_max)
        self.last_diagnostics={"settings":asdict(s),"added_group_delay_samples":-s.l_min,
            "fir_length":s.l_max-s.l_min+1,"captured_independent_path_energy_mean":float(captured.mean()),
            "captured_independent_path_energy_min":float(captured.min()),
            "per_realization_normalization":False}
        return taps
