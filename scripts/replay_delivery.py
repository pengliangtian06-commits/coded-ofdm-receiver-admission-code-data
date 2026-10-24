"""Read-only replay of frozen evidence and small CPU checkpoint/PHY checks.

Does not refit or overwrite an experiment. The output path must be new.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptive_ofdm.admission import NAMES, llrs, receive
from adaptive_ofdm.phy import Numerology, expected_frame_energy, transmit
from adaptive_ofdm.ldpc import NRLDPC
from run_admission_study import loadmodels, routers


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    study=ROOT/'runs/20261023-admission-study-v2'
    freeze=json.loads((study/'FROZEN.json').read_text());checks={}
    for group in ['source_sha256','weights_sha256','selection_sha256']:
        bad=[name for name,value in freeze[group].items() if digest(ROOT/name)!=value]
        if bad: raise RuntimeError({'hash_mismatches':bad})
        checks[group]={'verified_files':len(freeze[group]),'mismatches':bad}
    e=[]; choices=[]; neural=[]; transfer={}
    for seed in [1729,1730,1731]:
        d=np.load(study/f'seed{seed}/final.npz');c=np.load(study/f'seed{seed}/choices.npz')
        if d['errors'].shape!=(147456,9) or not np.isin(d['errors'],[0,1]).all():raise RuntimeError('invalid main outcome array')
        e.append(d['errors']);choices.append({k:c[k] for k in c.files})
        neural.append(float(np.mean(c['physical__16']>=7)))
    outcomes={name:np.stack([x[:,i] for x in e]) for i,name in enumerate(NAMES)}
    for key in ['physical__16','physical_classical_only__16','no_memory__16','no_consistency__16']:
        outcomes[key]=np.stack([e[j][np.arange(len(e[j])),choices[j][key]] for j in range(3)])
    means={k:float(v.mean()) for k,v in outcomes.items()}
    expected={'physical__16':.24190040870949073,'deep':.2838677300347222,'physical_classical_only__16':.24537489149305555}
    for k,v in expected.items():
        if abs(means[k]-v)>1e-12:raise RuntimeError((k,means[k],v))
    timing={}
    for device,path in [('cpu','20261023-admission-timing-v2'),('mps','20261023-admission-timing-mps-v2')]:
        rows=list(csv.DictReader((ROOT/'runs'/path/'summary.csv').open()))
        timing[device]={k:float(np.mean([float(r['mean_ms']) for r in rows if r['method']==k])) for k in ['deep','independent_shallow','physical','physical_classical_only']}
    for control in ['wider','higher_rate']:
        vals={k:[] for k in ['independent_shallow','deep','physical__16']}
        for seed in [1729,1730,1731]:
            d=np.load(study/f'seed{seed}/{control}/final.npz');c=np.load(study/f'seed{seed}/{control}/choices.npz')
            for k in vals:
                v=d['errors'][np.arange(len(d['frame'])),c[k]] if '__' in k else d['errors'][:,NAMES.index(k)]
                vals[k].append(float(v.mean()))
        transfer[control]={k:float(np.mean(v)) for k,v in vals.items()}
    aligned={}
    for window in ['unaligned','compensated']:
        vals={k:[] for k in ['deep','independent_shallow','physical__16','classical_only']}
        for seed in [1729,1730,1731]:
            d=np.load(ROOT/f'runs/20261023-alignment-control-v1/seed{seed}.npz');mask=d['alignment']==window
            # Alignment choices column order is declared by the control source.
            for k in vals:
                if k in ['deep','independent_shallow']:v=d['errors'][:,NAMES.index(k)]
                else:v=d['errors'][np.arange(len(mask)),d['choices'][:,0 if k=='physical__16' else 3]]
                vals[k].append(float(v[mask].mean()))
        aligned[window]={k:float(np.mean(v)) for k,v in vals.items()}
    torch.set_num_threads(4);cfg=Numerology(symbols=4);codec=NRLDPC(160,320,12);rng=np.random.default_rng(882901)
    info=rng.integers(0,2,(2,160),dtype=np.uint8);coded=codec.encode(info)
    if not np.array_equal(codec.decode((1-2*coded.astype(float))*20),info):raise RuntimeError('LDPC sign or roundtrip failure')
    obs,_=transmit(coded,cfg,rng,18,information_bits=160,occupancy=.08,mean_burst_length=8,inr_db=15)
    smoke=[];settings=json.loads((study/'classical_settings.json').read_text())
    kw={k:settings[k] for k in ['threshold','prior_scale','temperature']}
    for seed in [1729,1730,1731]:
        models,_=loadmodels(seed,'cpu');out=llrs(obs,cfg,*models,**kw)
        if len(out)!=9 or any(v.shape!=(2,320) or not np.isfinite(v).all() for v in out.values()):raise RuntimeError('LLR smoke failure')
        folder=study/f'seed{seed}';spec=json.loads((folder/'router_models.json').read_text());cost=np.array(json.loads((folder/'costs.json').read_text())['mean_ms']);rr=routers(spec,cost,16)
        decoded,choice=receive(obs.waveform[:1],cfg,*models,codec,**kw,router=rr['physical'])
        if decoded.shape!=(1,160):raise RuntimeError('complete path shape failure')
        smoke.append({'seed':seed,'finite_all_nine_llr_paths':True,'complete_path_branch':int(choice)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'passed','scope':'stored-outcome replay plus small CPU checkpoint/interface smoke; not independent full-data regeneration','frozen_integrity':checks,'mixture_bler':means,'neural_fraction':float(np.mean(neural)),'complete_mean_ms':timing,'fixed_weight_transfer_bler':transfer,'alignment_bler':aligned,'expected_frame_energy':expected_frame_energy(cfg),'checkpoint_smoke':smoke,'ldpc_roundtrip':True}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
