"""Development gate comparison with frame-group splits and strong simple controls.

Nothing in this script establishes final-test performance. The diagnostic
dataset is exploratory. Final evaluation requires new untouched seeds/regimes.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from adaptive_ofdm.policy import outcome_labels,lean_feature_transform


def fit_logistic(x,y,classes,regularization=.01):
    mean=x.mean(0); scale=np.maximum(x.std(0),1e-6)
    z=np.c_[(x-mean)/scale,np.ones(len(x))]
    onehot=np.eye(classes)[y]
    def objective(flat):
        w=flat.reshape(z.shape[1],classes)
        logits=z@w
        p=softmax(logits,axis=-1)
        loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()
        penalty=w.copy();penalty[-1]=0
        loss+=regularization*.5*np.square(penalty).sum()
        grad=z.T@(p-onehot)/len(y)+regularization*penalty
        return loss,grad.ravel()
    result=minimize(objective,np.zeros(z.shape[1]*classes),jac=True,method="L-BFGS-B",options={"maxiter":500})
    if not result.success:
        raise RuntimeError(result.message)
    w=result.x.reshape(z.shape[1],classes)
    return {"mean":mean,"scale":scale,"weights":w}


def predict(model,x):
    return softmax(np.c_[(x-model["mean"])/model["scale"],np.ones(len(x))]@model["weights"],axis=-1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.dataset)
    x=data["features"].astype(float)
    s=data["shallow_error"].astype(bool);d=data["deep_error"].astype(bool)
    labels=outcome_labels(s,d)
    lean=lean_feature_transform(x[:,0],x[:,8])
    frame,reg=data["frame"],data["regime"]
    # Same payload/channel frames repeat across regimes/SNR: split on frame,
    # not on individual rows. Shift regime is excluded from fitting/calibration.
    split1=int((frame.max()+1)*.5);split2=int((frame.max()+1)*.75)
    train=(frame<split1)&(reg<2)
    val=(frame>=split1)&(frame<split2)&(reg<2)
    test=frame>=split2
    models={
        "benefit_only":fit_logistic(x[train],(labels[train]==1).astype(int),2),
        "signed_outcome":fit_logistic(x[train],labels[train],3),
        "error_probability":fit_logistic(x[train],s[train].astype(int),2),
        "entropy_benefit":fit_logistic(x[train,:1],(labels[train]==1).astype(int),2),
        "lean_benefit":fit_logistic(lean[train],(labels[train]==1).astype(int),2),
        "entropy_quadratic":fit_logistic(lean[train,:2],(labels[train]==1).astype(int),2),
    }
    def inputs(name):
        if name=="entropy_benefit": return x[:,:1]
        if name=="lean_benefit": return lean
        if name=="entropy_quadratic": return lean[:,:2]
        return x
    probs={name:predict(model,inputs(name)) for name,model in models.items()}
    scores={"entropy_threshold":x[:,0],
            "benefit_only":probs["benefit_only"][:,1],
            "signed_outcome":probs["signed_outcome"][:,1]-probs["signed_outcome"][:,2],
            "error_probability":probs["error_probability"][:,1],
            "entropy_benefit":probs["entropy_benefit"][:,1]}
    for name in ("lean_benefit","entropy_quadratic"):
        scores[name]=probs[name][:,1]
    # Non-monotone one-dimensional entropy lookup control: train bins and
    # empirical continuation benefit, not a deliberately weak single threshold.
    edges=np.unique(np.quantile(x[train,0],np.linspace(0,1,11)))
    bins=np.digitize(x[:,0],edges[1:-1])
    value=np.array([np.mean((s&~d)[train&(bins==b)])-np.mean((~s&d)[train&(bins==b)])
                    for b in range(len(edges)-1)])
    scores["entropy_lookup"]=value[bins]
    rows=[]
    for name,score in scores.items():
        for fraction in [0,.05,.1,.2,.3,.4,.5,.6,.75,1]:
            threshold=np.inf if fraction==0 else (-np.inf if fraction==1 else np.quantile(score[val],1-fraction))
            decision=score>threshold
            for regime in np.unique(reg):
                mask=test&(reg==regime)
                selected=np.where(decision,d,s)
                rows.append({"method":name,"target_validation_continuation":fraction,"regime":int(regime),
                    "threshold":float(threshold),"frames":int(mask.sum()),"shallow_bler":float(s[mask].mean()),
                    "deep_bler":float(d[mask].mean()),"selected_bler":float(selected[mask].mean()),
                    "continuation_fraction":float(decision[mask].mean()),
                    "captured_beneficial":int(((labels==1)&decision&mask).sum()),
                    "incurred_harmful":int(((labels==2)&decision&mask).sum())})
    with (args.output/"frontiers.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(args.output/"scores.npz",**scores,shallow_error=s,deep_error=d,regime=reg,frame=frame,test=test)
    serializable={n:{k:v.tolist() for k,v in m.items()} for n,m in models.items()}
    (args.output/"models.json").write_text(json.dumps(serializable,indent=2))
    manifest={"status":"completed","purpose":"exploratory gate comparison; not final held-out evaluation",
        "dataset":str(args.dataset),"split_by":"frame group across all SNR/regimes",
        "training_frames_below":split1,"validation_frames_below":split2,
        "train_rows":int(train.sum()),"validation_rows":int(val.sum()),"test_rows":int(test.sum()),
        "train_label_counts":np.bincount(labels[train],minlength=3).tolist(),
        "entropy_lookup":{"edges":edges.tolist(),"benefit":value.tolist()},
        "cost_warning":"Continuation fraction is not total cost; profiling and head/gate/decoder overhead not included."}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for row in rows:
        if row["target_validation_continuation"]==.3:
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
