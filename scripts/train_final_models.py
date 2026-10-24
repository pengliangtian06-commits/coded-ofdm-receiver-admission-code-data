"""Train all six final controls with corrected fixed-pilot energy accounting."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=json.loads((ROOT/'configs/neural-strong-frontend-pilot.json').read_text())
for seed in [1729,1730,1731]:
    for kind in ['shallow','deep']:
        c=json.loads(json.dumps(base));c['seed']=seed
        c['purpose']='frozen final architecture; exact deterministic-pilot CP energy; independent '+kind+' control'
        if kind=='shallow':c['network'].update(blocks=2,exits=[2])
        config=ROOT/f'configs/final-{kind}-{seed}.json';config.write_text(json.dumps(c,indent=2))
        with (ROOT/f'logs/2026-10-23/final-{kind}-{seed}.log').open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'scripts/train_receiver.py'),'--config',str(config),'--output',str(ROOT/f'runs/20261023-final-{kind}-{seed}'),'--device','mps'],stdout=f,stderr=subprocess.STDOUT,check=True)
