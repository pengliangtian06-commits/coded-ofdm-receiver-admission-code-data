"""Train a modest multi-exit reference; preserve full configuration and losses."""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from adaptive_ofdm.phy import Numerology, transmit
from adaptive_ofdm.neural import MultiExitReceiver, features_from_grid, receiver_loss,receiver_inputs


def batch(rng, cfg, config, size):
    tr = config["training"]
    bits = rng.integers(0, 2, (size, cfg.coded_bits), dtype=np.uint8)
    # As in standard neural-demapper training, independent coded-bit targets
    # suffice; outer code is used for paired diagnostic evaluation only.
    obs, _ = transmit(bits, cfg, rng, rng.uniform(*tr["ebno_db"], size),
        information_bits=int(cfg.coded_bits*config["ldpc"]["rate"]),
        occupancy=float(rng.choice(tr["occupancies"])),
        mean_burst_length=float(rng.choice(tr["mean_lengths"])),
        inr_db=float(rng.uniform(*tr["inr_db"])))
    x, baseline = receiver_inputs(obs, cfg,config["network"].get("frontend","linear"))
    return x, baseline, bits.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/neural-pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cfg = Numerology(**config["numerology"])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/"src", args.output/"source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, args.output/"train_receiver.py")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(args.device)
    model = MultiExitReceiver(cfg, **config["network"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    # Validation is independently seeded and never used for gradient updates.
    validation_rng = np.random.default_rng(config["seed"]+100000)
    validation = [batch(validation_rng, cfg, config, 128) for _ in range(4)]
    manifest = {"status":"running", "config":config, "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "utc_start":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform":platform.platform(), "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
        "device":str(device), "parameters":sum(p.numel() for p in model.parameters()),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"src").rglob("*.py"))}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output/"config.json").write_text(json.dumps(config, indent=2))
    best, history = float("inf"), []
    start = time.perf_counter()
    for step in range(1, config["training"]["steps"]+1):
        x, base, target = [torch.from_numpy(v).to(device) for v in batch(rng,cfg,config,config["training"]["batch_size"])]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = receiver_loss(model(x, base), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % config["training"]["validation_interval"] == 0:
            model.eval()
            with torch.no_grad():
                val_loss = np.mean([receiver_loss(model(*[torch.from_numpy(v).to(device) for v in b[:2]]),
                                     torch.from_numpy(b[2]).to(device)).item() for b in validation])
            row = {"step":step,"loss":loss.item(),"validation_loss":float(val_loss),"elapsed_seconds":time.perf_counter()-start}
            history.append(row)
            print(json.dumps(row), flush=True)
            (args.output/"history.json").write_text(json.dumps(history, indent=2))
            if val_loss < best:
                best = val_loss
                torch.save({"state_dict":model.cpu().state_dict(),"config":config,"step":step}, args.output/"best.pt")
                model.to(device)
    manifest.update(status="completed", best_validation_loss=float(best), elapsed_seconds=time.perf_counter()-start)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
