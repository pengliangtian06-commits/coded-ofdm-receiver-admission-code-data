"""Predeclared admission study: tune, fit, freeze, then fresh final tests.

Every final condition is evaluated unchanged, including failure cases. All
model selection is restricted to separate development and validation draws.
"""
import argparse,csv,datetime,hashlib,json,platform,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptive_ofdm.phy import Numerology,transmit,demodulate
from adaptive_ofdm.channels import TDLSettings,TDLWaveformChannel
from adaptive_ofdm.neural import MultiExitReceiver
from adaptive_ofdm.classical import blank_waveform,delay_domain_lmmse_llr
from adaptive_ofdm.ldpc import NRLDPC
from adaptive_ofdm.admission import NAMES,admission_features,llrs,receive,LookupRouter
from adaptive_ofdm.routing import PredictiveRouter,select_branch
from fit_pilot_gates import fit_logistic,predict


def dump(path,value):path.write_text(json.dumps(value,indent=2))
def writecsv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadmodels(seed,device):
    shallow_path=ROOT/f'runs/20261023-final-shallow-{seed}/best.pt'
    deep_path=ROOT/f'runs/20261023-final-deep-{seed}/best.pt'
    models=[]
    for path in [shallow_path,deep_path]:
        state=torch.load(path,map_location='cpu',weights_only=False)
        m=MultiExitReceiver(Numerology(**state['config']['numerology']),**state['config']['network'])
        m.load_state_dict(state['state_dict']);m.to(device).eval();models.append(m)
    return models,[str(shallow_path.relative_to(ROOT)),str(deep_path.relative_to(ROOT))]


def makechannel(channel,seed):
    if channel['profile'] is None:return None
    return TDLWaveformChannel(TDLSettings(profile=channel['profile'],speed_m_per_s=channel['speed'],
        delay_spread_s=channel['delay_ns']*1e-9,seed=seed))


def batches(protocol,cfg,channel,regime,ebno,frames,seed,rate=.5):
    rng=np.random.default_rng(seed);ch=makechannel(channel,seed+100000)
    codec=NRLDPC(int(cfg.coded_bits*rate),cfg.coded_bits,12)
    for offset in range(0,frames,protocol['batch_size']):
        n=min(protocol['batch_size'],frames-offset)
        info=rng.integers(0,2,(n,codec.k),dtype=np.uint8);coded=codec.encode(info)
        taps=None if ch is None else ch(n,cfg.symbols*(cfg.fft_size+cfg.cp_length))
        obs,_=transmit(coded,cfg,rng,ebno,information_bits=codec.k,taps=taps,
            occupancy=regime['occupancy'],mean_burst_length=regime['length'],inr_db=regime['inr_db'])
        yield offset,info,coded,obs,codec,None if ch is None else ch.last_diagnostics


def tune(protocol,cfg,out):
    samples=[]
    # 3 channels x 2 regimes x 4 SNRs x 64 = 1536 independent validation frames.
    for ci,ch in enumerate(protocol['development_channels']):
        for reg in protocol['development_regimes']:
            for eb in protocol['ebno_db']:
                samples.extend(batches(protocol,cfg,ch,reg,eb,64,protocol['validation_seed']+ci*10000))
    rows=[];best=(np.inf,None)
    for threshold in protocol['blank_threshold_grid']:
        for scale in protocol['prior_scale_grid']:
            prior=np.exp(-np.arange(5)/2);prior*=scale/prior.sum();err=0;count=0;soft=[];targets=[]
            for _,info,coded,obs,codec,_ in samples:
                grid=demodulate(blank_waveform(obs.waveform,threshold),cfg)
                v=delay_domain_lmmse_llr(grid,cfg,True,prior,True)
                err+=int(np.any(codec.decode(v)!=info,axis=-1).sum());count+=len(info)
                soft.append(delay_domain_lmmse_llr(grid,cfg,True,prior));targets.append(coded)
            row={'threshold':threshold,'prior_scale':scale,'frames':count,'posterior_errors':err,'posterior_bler':err/count};rows.append(row)
            if err/count<best[0]:best=(err/count,(threshold,scale,np.concatenate(soft),np.concatenate(targets)))
    threshold,scale,v,b=best[1];sign=1-2*b.astype(float)
    fit=minimize_scalar(lambda t:np.logaddexp(0,-sign*v*t).mean(),bounds=(.1,4),method='bounded')
    if not fit.success:raise RuntimeError(fit.message)
    settings={'threshold':threshold,'prior_scale':scale,'temperature':float(fit.x),'selection':'minimum pooled posterior BLER on development-validation draws; ties follow declared grid order','temperature_objective':'coded-bit CE at selected settings','validation_seed':protocol['validation_seed'],'frames':count,'grid':rows}
    dump(out/'classical_settings.json',settings);print(json.dumps({'tuning':settings}),flush=True)
    return settings


def dataset(protocol,cfg,models,settings,seed,frames,channels,regimes,out,final=False,rate=.5):
    allx=[];alle=[];meta=[];diagnostics=[]
    for ci,ch in enumerate(channels):
        for reg in regimes:
            if final and ch['name']!='EXP5' and reg['name'] not in ['clean','matched','long_strong']:continue
            for eb in protocol['ebno_db']:
                nseen=0;counts=np.zeros(len(NAMES),int)
                for offset,info,_,obs,codec,diag in batches(protocol,cfg,ch,reg,eb,frames,seed+ci*10000,rate):
                    x=admission_features(obs,cfg);outputs=llrs(obs,cfg,*models,**{k:settings[k] for k in ['threshold','prior_scale','temperature']})
                    e=np.stack([np.any(codec.decode(outputs[i])!=info,axis=-1) for i in range(len(NAMES))],axis=-1)
                    allx.append(x);alle.append(e);counts+=e.sum(0);nseen+=len(e)
                    meta.extend([(ch['name'],reg['name'],eb,offset+i) for i in range(len(e))])
                    if diag is not None:diagnostics.append({'channel':ch['name'],**diag})
                print(json.dumps({'dataset':str(out),'channel':ch['name'],'regime':reg['name'],'ebno':eb,'n':nseen,'bler':(counts/nseen).tolist()}),flush=True)
    np.savez_compressed(out,features=np.concatenate(allx),errors=np.concatenate(alle),channel=np.array([m[0] for m in meta]),regime=np.array([m[1] for m in meta]),ebno=np.array([m[2] for m in meta]),frame=np.array([m[3] for m in meta]),branches=np.array(NAMES))
    if diagnostics:
        grouped={}
        for name in sorted(set(d['channel'] for d in diagnostics)):
            subset=[d for d in diagnostics if d['channel']==name]
            grouped[name]={'settings':subset[0]['settings'],'mean_captured_energy':float(np.mean([d['captured_independent_path_energy_mean'] for d in subset])),'minimum_batch_captured_energy':float(min(d['captured_independent_path_energy_min'] for d in subset)),'added_group_delay_samples':subset[0]['added_group_delay_samples'],'fir_length':subset[0]['fir_length']}
        dump(out.with_suffix('.channel.json'),grouped)


def profile(protocol,cfg,models,settings,out,frames=128):
    models=[m.to('cpu') for m in models];codec=NRLDPC(160,cfg.coded_bits,12)
    rng=np.random.default_rng(protocol['timing_seeds'][0]);bits=rng.integers(0,2,(frames,160),dtype=np.uint8)
    obs,_=transmit(codec.encode(bits),cfg,rng,rng.uniform(8,20,frames),information_bits=160,occupancy=.08,mean_burst_length=8,inr_db=15)
    rows=[]
    for _ in range(5):
        for k in range(len(NAMES)):receive(obs.waveform[:1],cfg,*models,codec,**{q:settings[q] for q in ['threshold','prior_scale','temperature']},choice=k)
    for i in range(frames):
        for k in rng.permutation(len(NAMES)):
            t=time.perf_counter_ns();receive(obs.waveform[i:i+1],cfg,*models,codec,**{q:settings[q] for q in ['threshold','prior_scale','temperature']},choice=k)
            rows.append({'frame':i,'branch':NAMES[k],'milliseconds':(time.perf_counter_ns()-t)/1e6})
    costs=[float(np.mean([r['milliseconds'] for r in rows if r['branch']==n])) for n in NAMES]
    writecsv(out/'cost_profile.csv',rows);dump(out/'costs.json',{'names':NAMES,'mean_ms':costs,'frames':frames,'scope':'full waveform-to-decoded-bit CPU batch1','platform':platform.platform(),'processor':platform.processor(),'torch_threads':torch.get_num_threads()})
    return np.array(costs)


def fit(protocol,trainpath,valpath,costs,out):
    tr=np.load(trainpath);va=np.load(valpath);x=tr['features'];v=va['features'];y=tr['errors'];yv=va['errors']
    specs={};report=[]
    for name,cols in {'physical':list(range(12)),'entropy':[0,1],'no_memory':[0,1,2,3,5,6,8,9,10,11],'no_consistency':list(range(8))}.items():
        models=[]
        for k in range(len(NAMES)):
            best=(np.inf,None,None)
            for reg in protocol['logistic_regularization_grid']:
                m=fit_logistic(x[:,cols],y[:,k].astype(int),2,reg);p=predict(m,v[:,cols])[:,1]
                ce=float(-np.mean(yv[:,k]*np.log(np.maximum(p,1e-12))+(1-yv[:,k])*np.log(np.maximum(1-p,1e-12))))
                if ce<best[0]:best=(ce,m,reg)
            models.append({q:a.tolist() for q,a in best[1].items()});report.append({'method':name,'branch':NAMES[k],'regularization':best[2],'validation_CE':best[0]})
        specs[name]={'columns':cols,'models':models}
    edges=np.unique(np.quantile(x[:,0],np.linspace(0,1,13)));bins=np.digitize(x[:,0],edges[1:-1]);table=np.stack([y[bins==j].mean(0) for j in range(len(edges)-1)])
    specs['entropy_lookup']={'edges':edges.tolist(),'table':table.tolist()}
    specs['fixed_risk']={'risk':y.mean(0).tolist()}
    dump(out/'router_models.json',specs);writecsv(out/'fit_validation.csv',report)
    return specs


def routers(specs,costs,penalty):
    result={}
    for name,spec in specs.items():
        if name=='fixed_risk':continue
        if name=='entropy_lookup':r=LookupRouter(spec['edges'],spec['table'],costs,penalty)
        else:r=PredictiveRouter(spec,costs,penalty);r.entropy_only=name=='entropy'
        result[name]=r
    classic=costs.copy();classic[7:]+=1e6
    result['physical_classical_only']=PredictiveRouter(specs['physical'],classic,penalty)
    return result


def evaluate(protocol,path,specs,costs,out):
    d=np.load(path);x=d['features'];e=d['errors'];rows=[];choices={}
    for penalty in protocol['penalties_ms']:
        for method,router in routers(specs,costs,penalty).items():choices[f'{method}__{penalty}']=router(x)
        choices[f'fixed_risk__{penalty}']=np.full(len(x),np.argmin(costs+penalty*np.asarray(specs['fixed_risk']['risk'])))
    for method,c in choices.items():
        selected=e[np.arange(len(e)),c]
        for ch in np.unique(d['channel']):
            for reg in np.unique(d['regime']):
                for eb in np.unique(d['ebno']):
                    mask=(d['channel']==ch)&(d['regime']==reg)&(d['ebno']==eb)
                    if not mask.any():continue
                    rows.append({'method':method,'channel':ch,'regime':reg,'ebno':int(eb),'frames':int(mask.sum()),'errors':int(selected[mask].sum()),'bler':float(selected[mask].mean()),'branch_cost_ms':float(costs[c[mask]].mean()),**{f'fraction_{n}':float((c[mask]==i).mean()) for i,n in enumerate(NAMES)}})
    writecsv(out/'frontiers.csv',rows);np.savez_compressed(out/'choices.npz',**choices)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=ROOT/'configs/admission-study-v1.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['develop','final'],required=True)
    a=p.parse_args();a.output=a.output.resolve();pr=json.loads(a.protocol.read_text());cfg=Numerology(**pr['numerology']);torch.set_num_threads(4)
    if a.stage=='develop':
        a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(a.protocol,a.output/'protocol.json')
        settings=tune(pr,cfg,a.output)
        for seed in pr['seeds']:
            out=a.output/f'seed{seed}';out.mkdir();models,paths=loadmodels(seed,'mps')
            dataset(pr,cfg,models,settings,pr['fit_seed'],pr['frames_per_development_point'],pr['development_channels'],pr['development_regimes'],out/'fitting.npz')
            # Tuning used first 64 frames at this seed. Use another seed here.
            dataset(pr,cfg,models,settings,pr['validation_seed']+500000,pr['frames_per_development_point']//2,pr['development_channels'],pr['development_regimes'],out/'validation.npz')
            costs=profile(pr,cfg,models,settings,out)
            specs=fit(pr,out/'fitting.npz',out/'validation.npz',costs,out)
            dump(out/'checkpoint_paths.json',paths)
        frozen={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':pr,'source_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted((ROOT/'src').rglob('*.py'))},'weights_sha256':{},'selection_warning':'final test must not change models, settings, risk calibration or operating points'}
        for seed in pr['seeds']:
            for path in json.loads((a.output/f'seed{seed}/checkpoint_paths.json').read_text()):frozen['weights_sha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        frozen['selection_sha256']={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.output/'classical_settings.json',*a.output.glob('seed*/router_models.json'),*a.output.glob('seed*/costs.json'),Path(__file__)]}
        dump(a.output/'FROZEN.json',frozen);print('DEVELOPMENT COMPLETE; final test not yet executed',flush=True)
    else:
        frozen=json.loads((a.output/'FROZEN.json').read_text())
        if frozen['protocol']!=pr:raise RuntimeError('protocol differs from freeze')
        for path,digest in {**frozen['source_sha256'],**frozen['weights_sha256'],**frozen['selection_sha256']}.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('source/weight changed after freeze: '+path)
        settings=json.loads((a.output/'classical_settings.json').read_text())
        for seed in pr['seeds']:
            out=a.output/f'seed{seed}';models,_=loadmodels(seed,'mps');costs=np.array(json.loads((out/'costs.json').read_text())['mean_ms']);specs=json.loads((out/'router_models.json').read_text())
            path=out/'final.npz'
            if path.exists():raise RuntimeError('refusing to overwrite final evaluation')
            dataset(pr,cfg,models,settings,pr['test_seed'],pr['frames_per_final_point'],pr['final_channels'],pr['final_regimes'],path,True)
            evaluate(pr,path,specs,costs,out)
            for name,control in [('wider','wider_control'),('higher_rate','higher_rate_control')]:
                spec=pr[control];cc=Numerology(**{**pr['numerology'],'symbols':spec['symbols']})
                folder=out/name;folder.mkdir()
                regs=[r for r in pr['final_regimes'] if r['name'] in spec['regimes']]
                dataset(pr,cc,models,settings,pr['test_seed']+200000,pr['frames_per_final_point'],[pr['final_channels'][0]],regs,folder/'final.npz',rate=spec['code_rate'])
                evaluate(pr,folder/'final.npz',specs,costs,folder)
        dump(a.output/'FINAL_COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest()})

if __name__=='__main__':main()
