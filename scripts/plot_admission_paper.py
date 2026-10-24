"""Publication figures from immutable final outcomes.

The burst-model illustration in Fig.2 is explicitly simulated and separately
seeded; no mock or demo results substitute for final receiver measurements.
"""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/figure_qa')]
from adaptive_ofdm.phy import Numerology,transmit,markov_bursts
from adaptive_ofdm.admission import NAMES
from adaptive_ofdm.routing import PredictiveRouter
from summarize_admission_final import interval

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})
COLORS={'linear':'#9B9B9B','blanked_isotropic':'#A27947','matched_coherent':'#999999','matched_posterior':'#597DA1',
        'matched_temperature':'#748A68','independent_shallow':'#7B69A2','deep':'#3A4148',
        'physical__16':'#007B80','entropy__16':'#B88A55','entropy_lookup__16':'#7F78A4','physical':'#007B80','entropy':'#B88A55','entropy_lookup':'#7F78A4'}
LABELS={'linear':'Pilot interpolation','blanked_isotropic':'Blanked isotropic','matched_coherent':'Matched coherent','matched_posterior':'Matched posterior',
        'matched_temperature':'Matched temperature','independent_shallow':'Independent shallow','deep':'Deep',
        'physical__16':'Physical admission','entropy__16':'Entropy admission','entropy_lookup__16':'Entropy lookup',
        'physical':'Physical admission','entropy':'Entropy admission','entropy_lookup':'Entropy lookup'}


def readcsv(path):return list(csv.DictReader(path.open()))
def writecsv(path,rows):
    with path.open('w') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def grid(titles,height=84):
    fig,axes=plt.subplots(1,len(titles),figsize=(183/25.4,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.085,right=.98,bottom=.21,top=.79,wspace=.36)
    for i,(ax,title) in enumerate(zip(axes[0],titles)):
        ax.set_title(title,pad=9)
        ax.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-27,9),textcoords='offset points',fontweight='bold',fontsize=9)
        ax.tick_params(width=.7,length=3)
    return fig,axes[0]
def legend(fig,ax,columns=3):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=columns,frameon=False,columnspacing=1.3,handlelength=1.8)
def save(fig,name,out,rows=None):
    path=out/name
    fig.canvas.draw()
    fig.savefig(str(path)+'.pdf',bbox_inches=None)
    fig.savefig(str(path)+'.svg',bbox_inches=None)
    fig.savefig(str(path)+'.png',dpi=600,bbox_inches=None)
    fig.savefig(str(path)+'.tiff',dpi=600,bbox_inches=None)
    if rows:writecsv(Path(str(path)+'.source.csv'),rows)
    plt.close(fig)


def setup_figures(out):
    fig,ax=plt.subplots(figsize=(183/25.4,82/25.4));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    boxes=[(.2,2.5,1.6,.85,'Received\nwaveform'),(2.2,2.5,1.65,.85,'FFT + observable\nphysical features'),(4.3,2.5,1.65,.85,'Risk + cost\nadmission'),(6.45,2.5,1.5,.85,'One selected\nsoft-bit path'),(8.35,2.5,1.4,.85,'One LDPC\ndecoder')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',linewidth=.8,edgecolor='#4B5A64',facecolor='#EAF1F3'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    for left,right in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(right[0]-.05,2.925),xytext=(left[0]+left[2]+.05,2.925),arrowprops={'arrowstyle':'->','color':'#4B5A64','lw':.8})
    ax.text(6.9,1.78,'Conventional paths  |  Shallow NN  |  Deep NN',ha='center',fontsize=8,color='#007B80')
    ax.text(5,.93,'Offline only: paired decoded errors fit path-risk models',ha='center',fontsize=8)
    ax.text(5,.48,'True bits, channel taps and interference states are excluded online',ha='center',fontsize=8,color='#5E6266')
    save(fig,'fig01_pipeline',out)
    cfg=Numerology(symbols=4);rng=np.random.default_rng(983721)
    obs,truth=transmit(rng.integers(0,2,(1,cfg.coded_bits),dtype=np.uint8),cfg,rng,12,information_bits=160,occupancy=.08,mean_burst_length=8,inr_db=15)
    fig,axes=grid(['Received waveform','Occupancy memory','Frame occupancy'],height=78)
    t=np.arange(obs.waveform.shape[1]);axes[0].plot(t,abs(obs.waveform[0]),color='#3A4148',lw=.7)
    axes[0].fill_between(t,0,float(abs(obs.waveform).max())*1.08,where=truth.burst_mask[0],color='#B88A55',alpha=.25,step='mid')
    axes[0].set(xlabel='Sample index',ylabel='Magnitude',xlim=(0,320),ylim=(0,float(abs(obs.waveform).max())*1.08))
    rows=[{'kind':'trace','sample_or_lag':int(i),'value':float(abs(obs.waveform[0,i])),'state':int(truth.burst_mask[0,i]),'regime':'Matched'} for i in t]
    for label,length,color in [('Independent',1/.92,'#9B9B9B'),('Matched',8,'#597DA1'),('Long',24,'#007B80')]:
        mask=markov_bursts(rng,4096,320,.08,length).astype(float);m=mask.mean();den=np.mean((mask-m)**2)
        lags=np.arange(25);corr=np.array([1.]+[np.mean((mask[:,lag:]-m)*(mask[:,:-lag]-m))/den for lag in lags[1:]])
        theory=(1-1/length-.08/(.92*length))**lags
        axes[1].plot(lags,corr,color=color,label=label,lw=1.4);axes[1].plot(lags,theory,color=color,ls=':',lw=.7)
        rows.extend({'kind':'correlation','sample_or_lag':int(lag),'value':float(value),'state':float(theory[lag]),'regime':label} for lag,value in zip(lags,corr))
        occupancy=np.sort(mask.mean(-1));axes[2].step(occupancy,np.arange(1,len(mask)+1)/len(mask),color=color,lw=1.3,where='post')
        rows.extend({'kind':'frame_occupancy','sample_or_lag':int(i),'value':float(value),'state':'','regime':label} for i,value in enumerate(occupancy))
    axes[2].set(xlabel='Occupied fraction',ylabel='Cumulative fraction',xlim=(0,1),ylim=(0,1.03))
    axes[1].set(xlabel='Lag (samples)',ylabel='State autocorrelation',xlim=(0,24),ylim=(-.05,1.05));legend(fig,axes[1]);save(fig,'fig02_burst_memory',out,rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--statistics',type=Path,required=True);p.add_argument('--timings',type=Path);p.add_argument('--alignment',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--setup-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    setup_figures(a.output)
    if a.setup_only:return
    conditions=readcsv(a.statistics/'conditions.csv');rng=np.random.default_rng(978001)
    data=[np.load(a.study/f'seed{s}/final.npz') for s in [1729,1730,1731]];choices=[np.load(a.study/f'seed{s}/choices.npz') for s in [1729,1730,1731]]
    d=data[0];e=np.stack([q['errors'] for q in data]);n=2048
    outcomes={name:e[:,:,i].astype(float) for i,name in enumerate(NAMES)}
    for key in ['physical__16','entropy__16','entropy_lookup__16','physical_classical_only__16']:
        outcomes[key]=np.stack([data[j]['errors'][np.arange(len(d['frame'])),choices[j][key]] for j in range(3)]).astype(float)
    methods=['linear','blanked_isotropic','matched_posterior','independent_shallow','deep']
    def curve(ax,channel,regime,method,rows):
        points=[r for r in conditions if r['channel']==channel and r['regime']==regime and r['method']==method];points.sort(key=lambda r:int(r['ebno']));rows.extend(points)
        x=np.array([int(r['ebno']) for r in points]);y=np.array([float(r['mean_bler']) for r in points]);lo=np.array([float(r['conditional_low']) for r in points]);hi=np.array([float(r['conditional_high']) for r in points])
        ax.errorbar(x,y,yerr=[y-lo,hi-y],color=COLORS[method],label=LABELS[method],marker='o',ms=3,lw=1.2,capsize=2)
        ax.set(xlabel='Background Eb/N0 (dB)',ylabel='Decoded BLER',xticks=[8,12,16,20],ylim=(-.015,1.02))
    fig,axes=grid(['Nominal channel, Gaussian background','Nominal channel, matched bursts']);rows=[]
    for ax,reg in zip(axes,['clean','matched']):
        for method in methods:curve(ax,'EXP5',reg,method,rows)
    legend(fig,axes[0]);save(fig,'fig03_strong_baselines',a.output,rows)
    fig,axes=grid(['Nominal channel, matched bursts','TDL-C, long/strong bursts'],height=89);rows=[]
    for ax,ch,reg in zip(axes,['EXP5','TDLC'],['matched','long_strong']):
        mask=(d['channel']==ch)&(d['regime']==reg);labels=['Coherent→posterior','Posterior→shallow','Shallow→deep']
        for j,(ref,alt) in enumerate([('matched_coherent','matched_posterior'),('matched_posterior','independent_shallow'),('independent_shallow','deep')]):
            r=outcomes[ref][:,mask];s=outcomes[alt][:,mask];cor=float(np.mean((r==1)&(s==0)));harm=float(np.mean((r==0)&(s==1)))
            cunit=np.mean((r==1)&(s==0),axis=0).reshape(-1,n).mean(0);hunit=np.mean((r==0)&(s==1),axis=0).reshape(-1,n).mean(0);clo,chi=interval(cunit,rng);hlo,hhi=interval(hunit,rng)
            ax.bar(j-.16,cor,width=.3,color='#597DA1',label='Corrections' if j==0 else None,yerr=[[cor-clo],[chi-cor]],capsize=2,error_kw={'elinewidth':.7})
            ax.bar(j+.16,harm,width=.3,color='#B88A55',label='Harms' if j==0 else None,yerr=[[harm-hlo],[hhi-harm]],capsize=2,error_kw={'elinewidth':.7})
            rows.append({'channel':ch,'regime':reg,'reference':ref,'alternative':alt,'correction_fraction':cor,'correction_low':clo,'correction_high':chi,'harm_fraction':harm,'harm_low':hlo,'harm_high':hhi,'predictions':int(r.size)})
        ax.set(xticks=np.arange(3),xticklabels=labels,ylabel='Paired transition fraction',ylim=(0,max(.08,max(q['correction_fraction'] for q in rows)*1.15)))
        ax.tick_params(axis='x',labelrotation=18)
        for label in ax.get_xticklabels():label.set_rotation_mode('anchor');label.set_ha('right')
    legend(fig,axes[0],2);save(fig,'fig04_correction_harm',a.output,rows)
    fig,axes=grid(['Conditions used during development','Excluded TDL-D profile']);rows=[]
    for ax,scope in zip(axes,['development','TDLD']):
        for method in ['physical','entropy','entropy_lookup']:
            means=[];spread=[];xx=[]
            for penalty in [0,1,2,4,8,16,32,64,128]:
                entries=[]
                for seed in [1729,1730,1731]:
                    rr=readcsv(a.study/f'seed{seed}/frontiers.csv')
                    selected=[r for r in rr if r['method']==f'{method}__{penalty}' and ((r['channel']=='TDLD') if scope=='TDLD' else (r['channel'] in ['EXP5','TDLA','TDLB'] and r['regime'] in ['clean','matched']))]
                    entries.append([np.mean([float(r['branch_cost_ms']) for r in selected]),np.mean([float(r['bler']) for r in selected])]);rows.extend(selected)
                entries=np.asarray(entries);xx.append(entries[:,0].mean());means.append(entries[:,1].mean());spread.append([entries[:,1].min(),entries[:,1].max()])
            ax.plot(xx,means,marker='o',ms=3,lw=1.4,color=COLORS[method],label=LABELS[method]);spread=np.array(spread)
            ax.vlines(xx,spread[:,0],spread[:,1],color=COLORS[method],lw=.8)
            ax.plot(xx[5],means[5],marker='*',ms=8,color=COLORS[method],markeredgecolor='white',markeredgewidth=.4)
        ax.set(xlabel='Estimated selected-branch cost (ms)',ylabel='Mean decoded BLER')
    legend(fig,axes[0]);save(fig,'fig05_branch_cost_frontier',a.output,rows)
    fig,axes=grid(['Duration and INR varied separately','Admission minus matched posterior']);rows=[]
    regimes=['matched','long_only','strong_only','long_strong'];labels=['L8 / 15dB','L24 / 15dB','L8 / 20dB','L24 / 20dB']
    for method in ['matched_posterior','independent_shallow','deep','physical__16']:
        means=[];lo=[];hi=[]
        for reg in regimes:
            mask=(d['channel']=='EXP5')&(d['regime']==reg);unit=outcomes[method][:,mask].mean(0).reshape(-1,n).mean(0);ci=interval(unit,rng)
            means.append(unit.mean());lo.append(ci[0]);hi.append(ci[1]);rows.append({'method':method,'regime':reg,'mean':float(unit.mean()),'low':ci[0],'high':ci[1]})
        y=np.array(means);axes[0].errorbar(np.arange(4),y,yerr=[y-np.array(lo),np.array(hi)-y],color=COLORS[method],label=LABELS[method],marker='o',lw=1.2,capsize=2,ms=3)
    for j,reg in enumerate(regimes):
        mask=(d['channel']=='EXP5')&(d['regime']==reg);unit=(outcomes['physical__16']-outcomes['matched_posterior'])[:,mask].mean(0).reshape(-1,n).mean(0);ci=interval(unit,rng,difference=True)
        axes[1].errorbar(j,unit.mean(),yerr=[[unit.mean()-ci[0]],[ci[1]-unit.mean()]],color='#007B80',marker='o',ms=4,capsize=3);rows.append({'method':'paired_difference','regime':reg,'mean':float(unit.mean()),'low':ci[0],'high':ci[1]})
    axes[1].axhline(0,color='#999999',ls=':',lw=.8)
    for ax in axes:ax.set(xticks=np.arange(4),xticklabels=labels);ax.tick_params(axis='x',labelrotation=15)
    axes[0].set_ylabel('Mean decoded BLER');axes[1].set_ylabel('Paired BLER difference');legend(fig,axes[0],2);save(fig,'fig06_duration_inr',a.output,rows)
    fig,axes=grid(['Combined shift at 20 dB','Neural use','Known-delay control'] if a.alignment else ['Combined shift at 20 dB','Neural use']);rows=[];channels=['EXP5','TDLA','TDLB','TDLC','TDLD']
    for method in ['blanked_isotropic','matched_posterior','independent_shallow','deep','physical__16']:
        means=[];low=[];high=[]
        for ch in channels:
            r=next(r for r in conditions if r['channel']==ch and r['regime']=='long_strong' and r['ebno']=='20' and r['method']==method);rows.append(r);means.append(float(r['mean_bler']));low.append(float(r['conditional_low']));high.append(float(r['conditional_high']))
        y=np.array(means);axes[0].errorbar(range(5),y,yerr=[y-low,np.array(high)-y],color=COLORS[method],label=LABELS[method],marker='o',ms=3,lw=1.2,capsize=2)
    for j,ch in enumerate(channels):
        mask=(d['channel']==ch)&(d['regime']=='long_strong')&(d['ebno']==20)
        vals=np.stack([q['physical__16'][mask]>=7 for q in choices]).mean(0);ci=interval(vals,rng)
        axes[1].errorbar(j,vals.mean(),yerr=[[vals.mean()-ci[0]],[ci[1]-vals.mean()]],marker='o',color='#007B80',capsize=3,ms=4)
        rows.append({'channel':ch,'regime':'long_strong','ebno':'20','method':'neural_fraction','mean_bler':float(vals.mean()),'conditional_low':ci[0],'conditional_high':ci[1]})
    for ax in axes[:2]:ax.set(xticks=range(5),xticklabels=channels)
    if a.alignment:
        control=[np.load(a.alignment/f'seed{s}.npz') for s in [1729,1730,1731]]
        for method in ['blanked_isotropic','matched_posterior','deep','physical__16']:
            y=[];low=[];high=[]
            for alignment in ['unaligned','compensated']:
                vals=[]
                for q in control:
                    mask=(q['channel']=='TDLC')&(q['regime']=='long_strong')&(q['ebno']==20)&(q['alignment']==alignment)
                    if method=='physical__16':vals.append(q['errors'][np.arange(len(q['frame'])),q['choices'][:,0]][mask])
                    else:vals.append(q['errors'][mask,NAMES.index(method)])
                unit=np.mean(vals,axis=0);ci=interval(unit,rng);y.append(unit.mean());low.append(ci[0]);high.append(ci[1]);rows.append({'channel':'TDLC','regime':'long_strong','ebno':'20','method':method,'alignment':alignment,'mean_bler':float(unit.mean()),'conditional_low':ci[0],'conditional_high':ci[1]})
            y=np.array(y);axes[2].errorbar([0,1],y,yerr=[y-low,np.array(high)-y],color=COLORS[method],marker='o',ms=3,lw=1.2,capsize=2)
        axes[2].set(xticks=[0,1],xticklabels=['Original','Compensated'],ylabel='TDL-C decoded BLER',xlim=(-.15,1.15))
    axes[0].set_ylabel('Decoded BLER');axes[1].set(ylabel='Fraction selecting a neural path',ylim=(-.02,1.03));legend(fig,axes[0]);save(fig,'fig07_channel_mismatch',a.output,rows)
    fig,axes=grid(['Development-profile final frames','Excluded-profile final frames']);rows=[]
    for ax,scope in zip(axes,['dev','TDLD']):
        mask=(d['channel']=='TDLD') if scope=='TDLD' else np.isin(d['channel'],['EXP5','TDLA','TDLB'])&np.isin(d['regime'],['clean','matched'])
        for method in ['physical','entropy']:
            pred=[];actual=[]
            for j,seed in enumerate([1729,1730,1731]):
                spec=json.loads((a.study/f'seed{seed}/router_models.json').read_text())[method];cost=np.array(json.loads((a.study/f'seed{seed}/costs.json').read_text())['mean_ms']);router=PredictiveRouter(spec,cost,16)
                risks=router.predict_risks(d['features'][mask]);choice=choices[j][method+'__16'][mask]
                pred.extend(risks[np.arange(len(choice)),choice]);actual.extend(data[j]['errors'][mask][np.arange(len(choice)),choice])
            pred=np.array(pred);actual=np.array(actual);px=[];py=[]
            for left in np.arange(0,1,.1):
                take=(pred>=left)&(pred<left+.1)
                if take.sum()<50:continue
                px.append(float(pred[take].mean()));py.append(float(actual[take].mean()));rows.append({'scope':scope,'method':method,'bin_left':left,'mean_risk':px[-1],'observed_error':py[-1],'predictions':int(take.sum())})
            ax.plot(px,py,marker='o',ms=3,lw=1.2,color=COLORS[method],label=LABELS[method])
        ax.plot([0,1],[0,1],color='#AAAAAA',ls=':',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted selected-path error risk',ylabel='Observed decoded-error fraction')
    legend(fig,axes[0],2);save(fig,'fig08_risk_calibration',a.output,rows)
    fig,axes=grid(['Eight-symbol frames, rate 0.5','Four-symbol frames, rate 0.75']);rows=[]
    for ax,control in zip(axes,['wider','higher_rate']):
        dd=[np.load(a.study/f'seed{s}/{control}/final.npz') for s in [1729,1730,1731]];cc=[np.load(a.study/f'seed{s}/{control}/choices.npz') for s in [1729,1730,1731]]
        for method in ['matched_posterior','independent_shallow','deep','physical__16']:
            y=[];low=[];high=[]
            for eb in [8,12,16,20]:
                mask=(dd[0]['regime']=='matched')&(dd[0]['ebno']==eb)
                vals=[]
                for j,q in enumerate(dd):
                    if '__' in method:vals.append(q['errors'][np.arange(len(mask)),cc[j][method]][mask])
                    else:vals.append(q['errors'][mask,NAMES.index(method)])
                unit=np.mean(vals,axis=0);ci=interval(unit,rng);y.append(unit.mean());low.append(ci[0]);high.append(ci[1]);rows.append({'control':control,'regime':'matched','method':method,'ebno':eb,'mean':float(unit.mean()),'low':ci[0],'high':ci[1]})
            y=np.array(y);ax.errorbar([8,12,16,20],y,yerr=[y-low,np.array(high)-y],color=COLORS[method],label=LABELS[method],marker='o',ms=3,lw=1.2,capsize=2)
        ax.set(xlabel='Background Eb/N0 (dB)',ylabel='Decoded BLER',xticks=[8,12,16,20],ylim=(-.02,1.03))
    legend(fig,axes[0],2);save(fig,'fig09_fixed_weight_transfer',a.output,rows)
    if a.timings:
        records=readcsv(a.timings/'timings.csv');summary=readcsv(a.timings/'summary.csv');fig,axes=grid(['Complete CPU path, batch one','Paired session/model mean difference']);rows=[]
        for method,style in zip(['matched_posterior','independent_shallow','deep','physical','entropy','entropy_lookup'],['-','--',':','-','-.',(0,(5,2,1,2))]):
            t=np.sort([float(r['milliseconds']) for r in records if r['method']==method]);axes[0].plot(t,np.arange(1,len(t)+1)/len(t),label=LABELS[method],color=COLORS[method],lw=1.2,linestyle=style)
        axes[0].set(xlabel='Waveform-to-decoded-bits latency (ms)',ylabel='Empirical cumulative fraction',ylim=(0,1),xlim=(0,np.quantile([float(r['milliseconds']) for r in records],.995)))
        for j,method in enumerate(['physical','entropy','entropy_lookup']):
            for seed in [1729,1730,1731]:
                for session in [0,1]:
                    r=next(r for r in summary if r['method']==method and int(r['seed'])==seed and int(r['session'])==session);ref=next(r for r in summary if r['method']=='deep' and r['seed']==str(seed) and r['session']==str(session));delta=float(r['mean_ms'])-float(ref['mean_ms']);axes[1].plot(j+(seed-1730)*.07,delta,marker='o' if session==0 else 's',ms=4,color=COLORS[method]);rows.append({'method':method,'seed':seed,'session':session,'delta_mean_ms':delta})
        axes[1].axhline(0,color='#999999',ls=':',lw=.8);axes[1].set(xticks=range(3),xticklabels=['Physical','Entropy','Lookup'],ylabel='Mean latency minus fixed deep (ms)');legend(fig,axes[0],3);save(fig,'fig10_complete_latency',a.output,rows)

if __name__=='__main__':main()
