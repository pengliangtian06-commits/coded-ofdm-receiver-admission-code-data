"""Observable classical controls with the same pooled-pilot preprocessing."""
import numpy as np
from .phy import demodulate
from .neural import features_from_grid


def pooled_pilot_llr(grid,cfg):
    """No network: use exactly the baseline LLR presented to neural exits."""
    _,base=features_from_grid(grid,cfg)
    data=np.flatnonzero(np.isin(cfg.active,cfg.data))
    return base[...,data].transpose(0,2,3,1).reshape(len(grid),-1)


def blanking_llr(waveform,cfg,threshold_multiplier=3.):
    """Conventional robust-amplitude threshold blanker followed by pooled LS.

    Threshold is fixed in advance, not chosen using test labels. It is a
    development control, not a reproduction of a particular recent method.
    """
    cleaned=blank_waveform(waveform,threshold_multiplier)
    return pooled_pilot_llr(demodulate(cleaned,cfg),cfg)


def blank_waveform(waveform,threshold_multiplier=3.):
    if threshold_multiplier<=0:
        raise ValueError("threshold multiplier must be positive")
    scale=np.maximum(np.median(abs(waveform),axis=-1)/np.sqrt(np.log(2)),1e-9)
    cleaned=np.where(abs(waveform)>threshold_multiplier*scale[:,None],0,waveform)
    return cleaned


def blanking_lmmse_llr(waveform,cfg,threshold_multiplier=3.,pool_symbols=False,tap_power=None):
    return delay_domain_lmmse_llr(demodulate(blank_waveform(waveform,threshold_multiplier),cfg),cfg,pool_symbols,tap_power)


def delay_domain_channel(grid,cfg,pool_symbols=False,tap_power=None,return_uncertainty=False):
    """Finite-delay Bayesian estimate with explicit tap statistics.

    Delay support uses CP length, not true channel taps. Noise scale is the
    observable null-subcarrier median. This is an additional classical control,
    not a reproduction of a particular paper or an oracle PDP estimator.
    """
    pilots=cfg.pilots%cfg.fft_size;data=cfg.data%cfg.fft_size
    null=np.setdiff1d(np.arange(cfg.fft_size),cfg.active%cfg.fft_size)
    n0=np.maximum(np.median(abs(grid[...,null])**2,axis=-1)/np.log(2),1e-8)
    nominal=n0.copy()
    observations=grid[...,pilots]
    if pool_symbols:
        observations=observations.mean(axis=1,keepdims=True)
        n0=np.broadcast_to(n0.mean(axis=1,keepdims=True)/cfg.symbols,observations.shape[:2])
    prior=np.full(cfg.cp_length,1/cfg.cp_length) if tap_power is None else np.asarray(tap_power,float)
    if prior.ndim!=1 or len(prior)>cfg.cp_length+1 or np.any(prior<=0):
        raise ValueError("tap powers must be positive and within CP support")
    length=len(prior)
    a=np.exp(-2j*np.pi*cfg.pilots[:,None]*np.arange(length)[None,:]/cfg.fft_size)
    design=a.conj().T@a
    rhs=np.einsum('lp,bsp->bsl',a.conj().T,observations)
    normal=design[None,None]+n0[...,None,None]*np.diag(1/prior)
    h_tap=np.linalg.solve(normal,rhs[...,None])[...,0]
    h=np.fft.fft(h_tap,n=cfg.fft_size,axis=-1)
    if return_uncertainty:
        covariance=np.linalg.inv(normal)*n0[...,None,None]
        f=np.exp(-2j*np.pi*np.arange(cfg.fft_size)[:,None]*np.arange(length)[None,:]/cfg.fft_size)
        variance=np.maximum(np.einsum('kl,bslm,km->bsk',f,covariance,f.conj(),optimize=True).real,0)
        return h,nominal,variance
    return h,nominal


def delay_domain_lmmse_llr(grid,cfg,pool_symbols=False,tap_power=None,account_uncertainty=False):
    if account_uncertainty:
        h,nominal,variance=delay_domain_channel(grid,cfg,pool_symbols,tap_power,True)
    else:h,nominal=delay_domain_channel(grid,cfg,pool_symbols,tap_power)
    data=cfg.data%cfg.fft_size
    h=h[...,data]
    y=grid[...,data]
    # Posterior-predictive Gaussian LLR includes channel variance when asked.
    # This is exact only for the assumed Gaussian prior/noise model; blanking
    # and burst interference break those assumptions. Nominal N0 is estimated.
    product=y*h.conj()
    denominator=nominal[...,None]+variance[...,data] if account_uncertainty else nominal[...,None]
    llr=np.stack([2*np.sqrt(2)*product.real/denominator,2*np.sqrt(2)*product.imag/denominator],axis=-1)
    return np.clip(llr,-20,20).reshape(len(grid),-1)
