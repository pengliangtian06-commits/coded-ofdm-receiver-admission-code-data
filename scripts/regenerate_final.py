"""Portable fresh evaluation of the hash-verified final models, without refit.

Original frozen runner is preserved. CPU replay can differ in floating point.
"""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptive_ofdm.phy import Numerology
from run_admission_study import loadmodels,dataset,evaluate,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps'],default='cpu');a=p.parse_args()
    original=ROOT/'runs/20261023-admission-study-v2';freeze=json.loads((original/'FROZEN.json').read_text());pr=freeze['protocol'];cfg=Numerology(**pr['numerology'])
    for group in ['source_sha256','weights_sha256','selection_sha256']:
        for name,sha in freeze[group].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('frozen artifact mismatch '+name)
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    settings=json.loads((original/'classical_settings.json').read_text());dump(a.output/'protocol.json',pr)
    dump(a.output/'RERUN_PROVENANCE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':'post-study full evaluation, no new fitting','neural_device':a.device,'original_freeze_sha256':hashlib.sha256((original/'FROZEN.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for seed in pr['seeds']:
        folder=a.output/f'seed{seed}';folder.mkdir();models,_=loadmodels(seed,a.device);src=original/f'seed{seed}'
        cost=np.array(json.loads((src/'costs.json').read_text())['mean_ms']);spec=json.loads((src/'router_models.json').read_text())
        dataset(pr,cfg,models,settings,pr['test_seed'],pr['frames_per_final_point'],pr['final_channels'],pr['final_regimes'],folder/'final.npz',True)
        evaluate(pr,folder/'final.npz',spec,cost,folder)
        for name,key in [('wider','wider_control'),('higher_rate','higher_rate_control')]:
            control=pr[key];cc=Numerology(**{**pr['numerology'],'symbols':control['symbols']});out=folder/name;out.mkdir()
            regs=[r for r in pr['final_regimes'] if r['name'] in control['regimes']]
            dataset(pr,cc,models,settings,pr['test_seed']+200000,pr['frames_per_final_point'],[pr['final_channels'][0]],regs,out/'final.npz',rate=control['code_rate'])
            evaluate(pr,out/'final.npz',spec,cost,out)
    dump(a.output/'FINAL_COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','role':'rerun of frozen evaluation','device':a.device})
if __name__=='__main__':main()
