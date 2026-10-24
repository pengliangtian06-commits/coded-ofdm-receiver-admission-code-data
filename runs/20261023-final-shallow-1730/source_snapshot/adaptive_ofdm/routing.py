"""Pre-neural receiver-selection diagnostics; generic routing is prior art.

Features use buffered received samples/pilots and classical soft information.
No neural output, transmitted bit, true channel or true noise input is used.
"""
import numpy as np
from .policy import binary_entropy_from_llr

FEATURE_NAMES=("classical_entropy","entropy_squared","log_pilot_variation",
               "log_time_kurtosis","time_power_correlation","log_null_variation",
               "entropy_x_kurtosis","kurtosis_x_correlation")
BRANCHES=("pooled_pilot","blanking","shallow","deep")
EXTENDED_BRANCHES=BRANCHES+("delay_lmmse_symbol","delay_lmmse_pool","blanking_lmmse","neural_frontend")


def pre_neural_features(observation,classical_llr,cfg):
    grid=observation.grid
    p=grid[...,cfg.pilots%cfg.fft_size]
    scale=np.maximum(np.mean(abs(grid[...,cfg.active%cfg.fft_size])**2,axis=(1,2)),1e-9)
    variation=np.mean(abs(p-p.mean(axis=1,keepdims=True))**2,axis=(1,2))/scale
    power=abs(observation.waveform)**2
    mean=np.maximum(power.mean(-1),1e-9)
    kurt=(power**2).mean(-1)/(mean*mean)
    memory=((power[:,1:]-mean[:,None])*(power[:,:-1]-mean[:,None])).mean(-1)
    memory/=np.maximum(power.var(-1),1e-9)
    null=np.setdiff1d(np.arange(cfg.fft_size),cfg.active%cfg.fft_size)
    null_power=(abs(grid[...,null])**2).mean(-1)/scale[:,None]
    entropy=binary_entropy_from_llr(classical_llr)
    logk=np.log(np.maximum(kurt,1))
    return np.stack([entropy,entropy**2,np.log1p(variation),logk,memory,
        np.log1p(null_power.std(-1)),entropy*logk,logk*memory],axis=-1)


def encode_joint_outcomes(errors):
    errors=np.asarray(errors)
    if errors.ndim!=2 or errors.shape[1]!=4 or not np.all((errors==0)|(errors==1)):
        raise ValueError("expected four paired binary error indicators")
    return (errors.astype(np.int64)*np.array([1,2,4,8])).sum(-1)


def marginal_error_risks(joint_probability):
    p=np.asarray(joint_probability,float)
    if p.ndim!=2 or p.shape[1]!=16 or np.any(p<0) or not np.allclose(p.sum(-1),1):
        raise ValueError("expected normalized 16-outcome probabilities")
    patterns=(np.arange(16)[:,None]>>np.arange(4))&1
    return p@patterns


def select_branch(risks,costs,error_penalty):
    risks=np.asarray(risks,float);costs=np.asarray(costs,float)
    if risks.ndim!=2 or costs.ndim!=1 or costs.size==0 or risks.shape[1]!=len(costs):
        raise ValueError("expected one cost for each action risk")
    if np.any(~np.isfinite(risks)) or np.any((risks<0)|(risks>1)) or np.any(~np.isfinite(costs)) or np.any(costs<0) or not np.isfinite(error_penalty) or error_penalty<0:
        raise ValueError("invalid decision inputs")
    return np.argmin(costs[None,:]+error_penalty*risks,axis=-1)


class PredictiveRouter:
    """Inference-only NumPy risk predictor; no outcomes/labels accepted."""
    def __init__(self,specification,costs,error_penalty):
        self.columns=specification["columns"]
        self.costs=np.asarray(costs)
        self.error_penalty=error_penalty
        def arrays(m):return {k:np.asarray(v) for k,v in m.items()}
        self.model=arrays(specification["model"]) if "model" in specification else None
        self.models=[arrays(m) for m in specification["models"]] if "models" in specification else None
        self.compact_weights=None
        if self.models and all(np.array_equal(m["mean"],self.models[0]["mean"]) and np.array_equal(m["scale"],self.models[0]["scale"]) for m in self.models):
            self.compact_weights=np.stack([m["weights"][:,1]-m["weights"][:,0] for m in self.models],axis=-1)

    @staticmethod
    def probabilities(model,x):
        z=np.c_[(x-model["mean"])/model["scale"],np.ones(len(x))]
        logits=z@model["weights"]
        weights=np.exp(logits-logits.max(axis=-1,keepdims=True))
        return weights/weights.sum(axis=-1,keepdims=True)

    def predict_risks(self,features):
        x=np.asarray(features)[:,self.columns]
        if self.model is not None:
            return marginal_error_risks(self.probabilities(self.model,x))
        if self.compact_weights is not None:
            m=self.models[0];z=np.c_[(x-m["mean"])/m["scale"],np.ones(len(x))]
            logits=z@self.compact_weights
            return np.exp(-np.logaddexp(0,-logits))
        return np.stack([self.probabilities(m,x)[:,1] for m in self.models],axis=-1)

    def __call__(self,features):
        return select_branch(self.predict_risks(features),self.costs,self.error_penalty)
