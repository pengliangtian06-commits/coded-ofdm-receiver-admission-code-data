"""Conditional paired-frame uncertainty; no tuning using final outcomes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptive_ofdm.admission import NAMES
from run_admission_study import dump,writecsv


def interval(values,rng,repeats=5000,difference=False):
    """Empirical scalar bootstrap via its discrete empirical distribution."""
    values=np.asarray(values,float);n=len(values)
    if np.all(values==0):
        upper=float(1-.05**(1/n))
        return [-upper,upper] if difference else [0.,upper]
    if not difference and np.all(values==1):return [float(.05**(1/n)),1.]
    levels,counts=np.unique(np.round(values,12),return_counts=True)
    samples=rng.multinomial(n,counts/counts.sum(),size=repeats)@levels/n
    return np.quantile(samples,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not (a.study/'FINAL_COMPLETE.json').exists():raise RuntimeError('final experiment incomplete')
    a.output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(970001)
    data=[np.load(a.study/f'seed{s}/final.npz') for s in [1729,1730,1731]]
    for d in data[1:]:
        for key in ['channel','regime','ebno','frame']:np.testing.assert_array_equal(data[0][key],d[key])
        np.testing.assert_array_equal(data[0]['errors'][:,:7],d['errors'][:,:7])
    choices=[np.load(a.study/f'seed{s}/choices.npz') for s in [1729,1730,1731]]
    d=data[0];e=np.stack([q['errors'] for q in data]);outcomes={name:e[:,:,i].astype(float) for i,name in enumerate(NAMES)}
    for method in choices[0].files:
        outcomes[method]=np.stack([data[j]['errors'][np.arange(len(d['frame'])),choices[j][method]] for j in range(3)]).astype(float)
    n=int(d['frame'].max()+1);rows=[];paired=[]
    def grouped(v,mask):
        frame=d['frame'][mask];values=v[:,mask].mean(0)
        np.testing.assert_array_equal(frame.reshape(-1,n),np.broadcast_to(np.arange(n),frame.reshape(-1,n).shape))
        return values.reshape(-1,n).mean(0)
    masks={'all':np.ones(len(d['frame']),bool),'development_conditions':np.isin(d['channel'],['EXP5','TDLA','TDLB'])&np.isin(d['regime'],['clean','matched']),
           'excluded_TDLD':d['channel']=='TDLD','transfer_TDLC':d['channel']=='TDLC','nominal_channel':d['channel']=='EXP5'}
    primary=['physical__16','no_memory__16','no_consistency__16','entropy__16','entropy_lookup__16','physical_classical_only__16','independent_shallow','deep','matched_posterior','blanked_isotropic']
    for scope,mask in masks.items():
        for method in primary:
            v=outcomes[method];byseed=v[:,mask].mean(1);unit=grouped(v,mask)
            lo,hi=interval(unit,rng)
            rows.append({'scope':scope,'method':method,'condition_evaluations_per_model':int(mask.sum()),'frame_groups':n,'mean_bler':float(unit.mean()),'conditional_low':lo,'conditional_high':hi,'seed_min':float(byseed.min()),'seed_max':float(byseed.max())})
        for ref in ['deep','independent_shallow','matched_posterior','blanked_isotropic','entropy__16','entropy_lookup__16','physical_classical_only__16','no_memory__16','no_consistency__16']:
            delta=outcomes['physical__16']-outcomes[ref];unit=grouped(delta,mask);lo,hi=interval(unit,rng,difference=True)
            paired.append({'scope':scope,'reference':ref,'delta_bler':float(unit.mean()),'paired_low':lo,'paired_high':hi,'conditional_on_models':3,'frame_groups':n})
    writecsv(a.output/'aggregate.csv',rows);writecsv(a.output/'paired_comparisons.csv',paired)
    conditions=[]
    for ch in np.unique(d['channel']):
        for reg in np.unique(d['regime']):
            for eb in np.unique(d['ebno']):
                mask=(d['channel']==ch)&(d['regime']==reg)&(d['ebno']==eb)
                if not mask.any():continue
                for method,v in outcomes.items():
                    if '__' in method and not method.endswith('__16'):continue
                    point=v[:,mask].mean(0);lo,hi=interval(point,rng)
                    conditions.append({'channel':ch,'regime':reg,'ebno':int(eb),'method':method,'frames_per_seed':int(mask.sum()),'errors_seed1729':int(v[0,mask].sum()),'errors_seed1730':int(v[1,mask].sum()),'errors_seed1731':int(v[2,mask].sum()),'mean_bler':float(point.mean()),'conditional_low':lo,'conditional_high':hi,'zero_error_one_sided_upper':float(1-.05**(1/n)) if point.max()==0 else None})
    writecsv(a.output/'conditions.csv',conditions)
    transitions=[]
    for ch in np.unique(d['channel']):
        for reg in np.unique(d['regime']):
            mask=(d['channel']==ch)&(d['regime']==reg)
            if not mask.any():continue
            for alternative,reference in [('deep','matched_posterior'),('independent_shallow','matched_posterior'),('deep','independent_shallow'),('physical__16','matched_posterior')]:
                ref=outcomes[reference][:,mask];alt=outcomes[alternative][:,mask]
                transitions.append({'channel':ch,'regime':reg,'alternative':alternative,'reference':reference,'paired_predictions':int(ref.size),'corrections':int(((ref==1)&(alt==0)).sum()),'harms':int(((ref==0)&(alt==1)).sum())})
    writecsv(a.output/'transitions.csv',transitions)
    dump(a.output/'manifest.json',{'status':'completed','bootstrap_seed':970001,'bootstrap_repeats':5000,'interval_scope':'conditional on three trained models; group identical frame index across conditions/seeds','zero_error':'one-sided exact Bernoulli upper at95%, not degenerate empirical bootstrap','main_condition_count':len(set(zip(d['channel'].tolist(),d['regime'].tolist(),d['ebno'].tolist()))),'frames_per_condition_per_seed':n,'models':3,'outcome_evaluations':int(e.size),'warning':'multiple frontiers and per-condition intervals descriptive, not multiplicity-adjusted confirmatory tests'})

if __name__=='__main__':main()
