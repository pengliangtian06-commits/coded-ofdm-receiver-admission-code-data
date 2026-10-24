"""Four fixed receiver paths with an explicit received-waveform boundary."""
import numpy as np
import torch
from .phy import demodulate,Observation
from .classical import pooled_pilot_llr,blanking_llr,delay_domain_lmmse_llr,blanking_lmmse_llr
from .neural import features_from_grid,receiver_inputs


def base_to_data(base,cfg):
    indices=np.flatnonzero(np.isin(cfg.active,cfg.data))
    return base[...,indices].transpose(0,2,3,1).reshape(len(base),-1)


def fixed_path_llr(waveform,cfg,model,branch):
    if branch==1:
        return blanking_llr(waveform,cfg)
    if branch==6:return blanking_lmmse_llr(waveform,cfg)
    grid=demodulate(waveform,cfg)
    if branch==7:
        obs=Observation(waveform,grid,np.full(len(grid),np.nan))
        _,base=receiver_inputs(obs,cfg,model.frontend)
        return base_to_data(base,cfg)
    if branch==0:
        return pooled_pilot_llr(grid,cfg)
    if branch in (4,5):return delay_domain_lmmse_llr(grid,cfg,pool_symbols=branch==5)
    obs=Observation(waveform,grid,np.full(len(grid),np.nan))
    x,base=[torch.from_numpy(v) for v in receiver_inputs(obs,cfg,model.frontend)]
    with torch.inference_mode():
        if branch==2:
            output,_=model.prefix(x,base)
        elif branch==3:
            output=model.final_only(x,base)
        else:
            raise ValueError("unknown receiver branch")
    return output.numpy()


def prepare_routing(waveform,cfg):
    grid=demodulate(waveform,cfg)
    x,base=features_from_grid(grid,cfg)
    obs=Observation(waveform,grid,np.full(len(grid),np.nan))
    return obs,x,base,base_to_data(base,cfg)


def adaptive_path_llr(waveform,cfg,model,router):
    """Batch-one routing with reused frontend; does not run unselected NNs."""
    from .routing import pre_neural_features
    if len(waveform)!=1:
        raise ValueError("batch-one runtime required for current benchmark")
    observation,x,base,classical=prepare_routing(waveform,cfg)
    features=pre_neural_features(observation,classical,cfg)
    branch=int(router(features)[0])
    if branch==0:return classical,branch
    if branch==1:return blanking_llr(waveform,cfg),branch
    if branch in (4,5):return delay_domain_lmmse_llr(observation.grid,cfg,pool_symbols=branch==5),branch
    if branch==6:return blanking_lmmse_llr(waveform,cfg),branch
    if branch==7:
        if model.frontend=="linear":return classical,branch
        _,base=receiver_inputs(observation,cfg,model.frontend)
        return base_to_data(base,cfg),branch
    if model.frontend!="linear":x,base=receiver_inputs(observation,cfg,model.frontend)
    with torch.inference_mode():
        x,base=torch.from_numpy(x),torch.from_numpy(base)
        if branch==2:output,_=model.prefix(x,base)
        else:output=model.final_only(x,base)
    return output.numpy(),branch
