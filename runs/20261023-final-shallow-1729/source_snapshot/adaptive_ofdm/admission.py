"""Frozen study bank and observable admission features.

This is an engineering comparison, not a claim of novel logistic regression,
Bayesian selection, blanking, neural residuals or model checking.
"""
import numpy as np
import torch
from .phy import demodulate,Observation
from .classical import pooled_pilot_llr,blank_waveform,delay_domain_lmmse_llr
from .neural import receiver_inputs
from .routing import pre_neural_features,select_branch
from .policy import binary_entropy_from_llr

NAMES=("linear","blanked_linear","isotropic","blanked_isotropic",
       "matched_coherent","matched_posterior","matched_temperature","independent_shallow","deep")


def admission_features(obs,cfg,entropy_only=False):
    base=pooled_pilot_llr(obs.grid,cfg)
    if entropy_only:
        entropy=binary_entropy_from_llr(base)
        result=np.zeros((len(base),12));result[:,0]=entropy;result[:,1]=entropy**2
        return result
    basic=pre_neural_features(obs,base,cfg)
    p=obs.grid[...,cfg.pilots%cfg.fft_size].mean(1)
    scale=np.maximum(np.mean(abs(p)**2,axis=-1),1e-9)
    rough=np.mean(abs(np.diff(p,axis=-1))**2,axis=-1)/scale
    # Fit alternate pilots to the nominal five-tap subspace, then predict
    # withheld pilots. No noise truth, training labels or NN outputs enter.
    indices=cfg.pilots;fit=np.arange(len(indices))%2==0;hold=~fit
    a=np.exp(-2j*np.pi*indices[:,None]*np.arange(5)[None,:]/cfg.fft_size)
    h=p[:,fit]@np.linalg.pinv(a[fit]).T
    residual=np.mean(abs(p[:,hold]-h@a[hold].T)**2,axis=-1)/scale
    return np.c_[basic,np.log1p(rough),np.log1p(residual),
                 basic[:,0]*np.log1p(residual),basic[:,3]*np.log1p(residual)]


def llrs(obs,cfg,shallow,deep,threshold=3.,prior_scale=1.,temperature=1.,only=None):
    """One selected path in runtime; all paths for offline paired labels."""
    wanted=set(range(len(NAMES))) if only is None else {int(only)}
    out={}
    if 0 in wanted:out[0]=pooled_pilot_llr(obs.grid,cfg)
    if wanted.intersection({1,3,4,5,6}):
        blanked=demodulate(blank_waveform(obs.waveform,threshold),cfg)
    if 1 in wanted:out[1]=pooled_pilot_llr(blanked,cfg)
    if 2 in wanted:out[2]=delay_domain_lmmse_llr(obs.grid,cfg,True,None,True)
    if 3 in wanted:out[3]=delay_domain_lmmse_llr(blanked,cfg,True,None,True)
    if wanted.intersection({4,5,6}):
        prior=np.exp(-np.arange(5)/2);prior*=prior_scale/prior.sum()
        if 4 in wanted:out[4]=delay_domain_lmmse_llr(blanked,cfg,True,prior)
        if 5 in wanted:out[5]=delay_domain_lmmse_llr(blanked,cfg,True,prior,True)
        if 6 in wanted:out[6]=delay_domain_lmmse_llr(blanked,cfg,True,prior)*temperature
    for k,model in [(7,shallow),(8,deep)]:
        if k in wanted:
            # NN weights keep their own fixed threshold-3 coherent frontend;
            # changing the tuned conventional path does not alter NN inputs.
            device=next(model.parameters()).device
            x,b=receiver_inputs(obs,cfg,model.frontend)
            with torch.inference_mode():
                out[k]=model.final_only(torch.from_numpy(x).to(device),torch.from_numpy(b).to(device)).cpu().numpy()
    return out


def receive(waveform,cfg,shallow,deep,codec,threshold,prior_scale,temperature=1.,choice=None,router=None):
    """Full waveform-to-decoded-bit operation, including all decision costs."""
    obs=Observation(waveform,demodulate(waveform,cfg),None)
    if router is not None:
        features=admission_features(obs,cfg,getattr(router,"entropy_only",False))
        choice=int(router(features)[0])
    if choice is None:raise ValueError("choose a fixed path or a router")
    output=llrs(obs,cfg,shallow,deep,threshold,prior_scale,temperature,only=choice)[choice]
    return codec.decode(output),choice


class LookupRouter:
    entropy_only=True
    def __init__(self,edges,table,costs,penalty):
        self.edges=np.asarray(edges);self.table=np.asarray(table)
        self.costs=np.asarray(costs);self.penalty=penalty
    def predict_risks(self,features):
        return self.table[np.digitize(features[:,0],self.edges[1:-1])]
    def __call__(self,features):
        return select_branch(self.predict_risks(features),self.costs,self.penalty)
