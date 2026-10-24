"""Two-session, randomized, full-path timing of frozen admission systems."""
import argparse,hashlib,json,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_admission_study import batches,loadmodels,routers,writecsv,dump
from adaptive_ofdm.phy import Numerology
from adaptive_ofdm.admission import NAMES,receive
from adaptive_ofdm.ldpc import NRLDPC


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    protocol=json.loads((a.study/'protocol.json').read_text());cfg=Numerology(**protocol['numerology'])
    settings=json.loads((a.study/'classical_settings.json').read_text());kw={k:settings[k] for k in ['threshold','prior_scale','temperature']}
    codec=NRLDPC(160,cfg.coded_bits,12);rows=[];summary=[]
    for seed in protocol['seeds']:
        folder=a.study/f'seed{seed}';models,_=loadmodels(seed,'cpu')
        costs=np.array(json.loads((folder/'costs.json').read_text())['mean_ms']);specs=json.loads((folder/'router_models.json').read_text())
        rr=routers(specs,costs,protocol['primary_penalty_ms'])
        methods={n:('fixed',i) for i,n in enumerate(NAMES)}
        methods.update({n:('router',rr[n]) for n in ['physical','entropy','entropy_lookup','physical_classical_only']})
        for session,drawseed in enumerate(protocol['timing_seeds']):
            waves=[];info=[];metadata=[]
            # Eight frames in each of the 72 predeclared main conditions.
            for ci,ch in enumerate(protocol['final_channels']):
                for reg in protocol['final_regimes']:
                    if ch['name']!='EXP5' and reg['name'] not in ['clean','matched','long_strong']:continue
                    for eb in protocol['ebno_db']:
                        for _,bits,_,obs,_,_ in batches(protocol,cfg,ch,reg,eb,8,drawseed+10000*ci):
                            waves.extend(obs.waveform[i:i+1] for i in range(len(bits)));info.extend(bits)
                            metadata.extend([(ch['name'],reg['name'],eb,i) for i in range(len(bits))])
            rng=np.random.default_rng(drawseed+seed)
            for _ in range(10):
                for name,(kind,method) in methods.items():receive(waves[0],cfg,*models,codec,**kw,**({'choice':method} if kind=='fixed' else {'router':method}))
            for i in rng.permutation(len(waves)):
                for name in rng.permutation(list(methods)):
                    kind,method=methods[name];start=time.perf_counter_ns()
                    decoded,choice=receive(waves[i],cfg,*models,codec,**kw,**({'choice':method} if kind=='fixed' else {'router':method}))
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    ch,reg,eb,frame=metadata[i]
                    rows.append({'seed':seed,'session':session,'frame_group':int(i),'channel':ch,'regime':reg,'ebno':eb,'method':name,'branch':NAMES[choice],'milliseconds':elapsed,'error':int(np.any(decoded[0]!=info[i]))})
            print(json.dumps({'seed':seed,'session':session,'timed_frames':len(waves),'methods':len(methods)}),flush=True)
    writecsv(a.output/'timings.csv',rows)
    for name in methods:
        for seed in protocol['seeds']:
            for session in [0,1]:
                r=[q for q in rows if q['method']==name and q['seed']==seed and q['session']==session];t=np.array([q['milliseconds'] for q in r])
                summary.append({'method':name,'seed':seed,'session':session,'frames':len(r),'mean_ms':float(t.mean()),'median_ms':float(np.median(t)),'p95_ms':float(np.quantile(t,.95)),'p99_ms':float(np.quantile(t,.99)),'bler':float(np.mean([q['error'] for q in r]))})
    writecsv(a.output/'summary.csv',summary)
    dump(a.output/'manifest.json',{'status':'completed','platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4,'sessions':2,'timing_frames_per_session_per_model':len(waves),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_warning':'macOS process-wide peak, not incremental path memory','scope':'complete waveform-to-decoded-bits including selection; batch1 CPU','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

if __name__=='__main__':main()
