"""Post-freeze, untuned mechanism control for a known FIR group delay.

Main results are retained. Both observations come from one extended received
waveform. This control distinguishes deterministic timing alignment from
propagation-prior mismatch; it is not an additional fitted robustness result.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptive_ofdm.phy import Numerology,make_grid,modulate,apply_channel,expected_frame_energy,markov_bursts,Observation,demodulate
from adaptive_ofdm.admission import NAMES,llrs,admission_features
from adaptive_ofdm.ldpc import NRLDPC
from run_admission_study import loadmodels,makechannel,routers,dump,writecsv


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    protocol=json.loads((a.study/'protocol.json').read_text());cfg=Numerology(**protocol['numerology']);settings=json.loads((a.study/'classical_settings.json').read_text());codec=NRLDPC(160,cfg.coded_bits,12)
    control={'purpose':'mechanism control added after freezing primary study; no refit, calibration or operating-point change','seed':991001,'frames_per_point':1024,'channels':['TDLA','TDLC','TDLD'],'regimes':['clean','matched','long_strong'],'ebno_db':[12,20],'alignment':'unaligned y[0:T] and compensated y[4:T+4] from the identical extended received waveform','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};dump(a.output/'PREDECLARED.json',control)
    rows=[]
    for modelseed in protocol['seeds']:
        models,_=loadmodels(modelseed,'mps');folder=a.study/f'seed{modelseed}';cost=np.array(json.loads((folder/'costs.json').read_text())['mean_ms']);spec=json.loads((folder/'router_models.json').read_text());rr=routers(spec,cost,16)
        saved=[];meta=[];choices=[]
        for ci,channel in enumerate([c for c in protocol['final_channels'] if c['name'] in control['channels']]):
            for reg in [r for r in protocol['final_regimes'] if r['name'] in control['regimes']]:
                for eb in control['ebno_db']:
                    rng=np.random.default_rng(control['seed']+ci*10000);ch=makechannel(channel,control['seed']+ci*10000+100000)
                    counts={key:np.zeros(len(NAMES),int) for key in ['unaligned','compensated']}
                    for offset in range(0,control['frames_per_point'],64):
                        n=64;info=rng.integers(0,2,(n,160),dtype=np.uint8);coded=codec.encode(info);x=modulate(make_grid(coded,cfg),cfg);T=x.shape[1]
                        extended=np.pad(x,((0,0),(0,4)));taps=ch(n,T+4);clean=apply_channel(extended,taps,cfg)
                        n0=expected_frame_energy(cfg)/160/10**(eb/10)
                        noise=(rng.standard_normal(extended.shape)+1j*rng.standard_normal(extended.shape))*np.sqrt(n0/2)
                        mask=markov_bursts(rng,n,T+4,reg['occupancy'],reg['length'])
                        impulse=(rng.standard_normal(extended.shape)+1j*rng.standard_normal(extended.shape))*np.sqrt(n0*10**(reg['inr_db']/10)/2)
                        y=clean+noise+mask*impulse
                        for alignment,start in [('unaligned',0),('compensated',4)]:
                            waveform=y[:,start:start+T];obs=Observation(waveform,demodulate(waveform,cfg),None);f=admission_features(obs,cfg)
                            outputs=llrs(obs,cfg,*models,**{k:settings[k] for k in ['threshold','prior_scale','temperature']})
                            errors=np.stack([np.any(codec.decode(outputs[k])!=info,axis=-1) for k in range(len(NAMES))],axis=-1)
                            counts[alignment]+=errors.sum(0);saved.append(errors);choices.append(np.stack([rr[name](f) for name in ['physical','entropy','entropy_lookup','physical_classical_only']],axis=-1))
                            meta.extend([(channel['name'],reg['name'],eb,alignment,offset+i) for i in range(n)])
                    for alignment,count in counts.items():
                        row={'seed':modelseed,'channel':channel['name'],'regime':reg['name'],'ebno':eb,'alignment':alignment,'frames':1024,**{name:float(count[i]/1024) for i,name in enumerate(NAMES)}};rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(a.output/f'seed{modelseed}.npz',errors=np.concatenate(saved),choices=np.concatenate(choices),channel=np.array([m[0] for m in meta]),regime=np.array([m[1] for m in meta]),ebno=np.array([m[2] for m in meta]),alignment=np.array([m[3] for m in meta]),frame=np.array([m[4] for m in meta]),branches=np.array(NAMES))
    writecsv(a.output/'summary.csv',rows);dump(a.output/'COMPLETE.json',{'status':'completed','note':'secondary timing-alignment mechanism control, not a newly tuned main test'})

if __name__=='__main__':main()
