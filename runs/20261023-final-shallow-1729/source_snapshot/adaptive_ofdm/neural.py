"""Multi-exit residual CNN reference and observation-only preprocessing.

Architecture is an experimental baseline, not an originality claim. Complete
frame processing is non-causal within a frame (a buffered receiver).
"""
import numpy as np
import torch
from torch import nn
from .phy import Numerology


def features_from_grid(grid, cfg: Numerology,frontend="linear"):
    """No transmitted-bit, true-channel, or nominal-noise inputs.

    Guard/DC energy estimates interference-plus-noise; its median estimator
    assumes Gaussian null energy and can be biased under bursts. This is a
    deliberately observable baseline, not a perfect noise-knowledge claim.
    """
    active = cfg.active % cfg.fft_size
    pilot = cfg.pilots % cfg.fft_size
    null = np.setdiff1d(np.arange(cfg.fft_size), active)
    scale = np.sqrt(np.maximum(np.mean(abs(grid[..., active])**2, axis=(1, 2)), 1e-8))
    y = grid / scale[:, None, None]
    if frontend=="linear":
        h_p = y[..., pilot].mean(axis=1)
        h = np.stack([np.interp(cfg.active, cfg.pilots, hp.real) +
                      1j*np.interp(cfg.active, cfg.pilots, hp.imag) for hp in h_p])[:,None]
        n0 = np.maximum(np.median(abs(y[..., null])**2, axis=(1, 2))/np.log(2), 1e-4)[:,None]
    elif frontend=="matched_pdp":
        from .classical import delay_domain_channel
        prior=np.exp(-np.arange(5)/2);prior/=prior.sum()
        h_raw,nominal=delay_domain_channel(grid,cfg,pool_symbols=True,tap_power=prior)
        h=h_raw[...,active]/scale[:,None,None]
        n0=np.maximum(nominal/scale[:,None]**2,1e-8)
    else:raise ValueError("unknown frontend")
    h=np.broadcast_to(h,(len(grid),cfg.symbols,len(active)))
    n0=np.broadcast_to(n0,(len(grid),cfg.symbols))
    received = y[..., active]
    z = received * h.conj() / np.maximum(abs(h)**2, 0.03)
    pilot_mask = np.broadcast_to(np.isin(cfg.active, cfg.pilots), h.shape)
    noise_map = np.broadcast_to(np.log(n0)[...,None], h.shape)
    features = np.stack([received.real, received.imag, h.real, h.imag,
                         np.clip(z.real, -8, 8), np.clip(z.imag, -8, 8),
                         pilot_mask, noise_map], axis=1).astype(np.float32)
    # Coherent QPSK log(P(0)/P(1)); no division by h is needed here.
    matched = received*h.conj()
    base = np.stack([2*np.sqrt(2)*matched.real/n0[...,None],
                     2*np.sqrt(2)*matched.imag/n0[...,None]], axis=1)
    return features, np.clip(base, -20, 20).astype(np.float32)


def receiver_inputs(observation,cfg,frontend="linear"):
    grid=observation.grid
    if frontend=="matched_pdp_blanked":
        from .classical import blank_waveform
        from .phy import demodulate
        grid=demodulate(blank_waveform(observation.waveform),cfg)
        frontend="matched_pdp"
    return features_from_grid(grid,cfg,frontend)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return torch.nn.functional.silu(x+self.net(x))


class MultiExitReceiver(nn.Module):
    def __init__(self, cfg=Numerology(symbols=4), width=24, blocks=4, exits=(1, 4),frontend="linear"):
        super().__init__()
        if not exits or sorted(set(exits)) != list(exits) or exits[-1] != blocks:
            raise ValueError("exits must be sorted unique and include final block")
        if exits[0] < 1:
            raise ValueError("exit indices are one based")
        self.cfg, self.exits = cfg, tuple(exits)
        self.frontend=frontend
        self.stem = nn.Conv2d(8, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        self.heads = nn.ModuleDict({str(i): nn.Conv2d(width, 2, 1) for i in exits})
        indices = np.flatnonzero(np.isin(cfg.active, cfg.data))
        self.register_buffer("data_indices", torch.tensor(indices, dtype=torch.long))
        # Initial output equals the classical observable baseline.
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def llr_from_head(self, x, base, depth):
        logits = base+self.heads[str(depth)](x)
        selected = logits.index_select(-1, self.data_indices)
        return selected.permute(0, 2, 3, 1).reshape(len(x), -1)

    def forward(self, features, base, max_depth=None):
        x = torch.nn.functional.silu(self.stem(features))
        out = []
        limit = len(self.blocks) if max_depth is None else max_depth
        for depth, block in enumerate(self.blocks, 1):
            x = block(x)
            if depth in self.exits:
                out.append(self.llr_from_head(x, base, depth))
            if depth >= limit:
                break
        return out

    def prefix(self, features, base):
        """Return shallow LLRs plus retained state without executing deep blocks."""
        x = torch.nn.functional.silu(self.stem(features))
        for block in self.blocks[:self.exits[0]]:
            x = block(x)
        return self.llr_from_head(x, base, self.exits[0]), x

    def continuation(self, state, base):
        """Reuse prefix state; do not recompute already executed blocks."""
        x = state
        for block in self.blocks[self.exits[0]:]:
            x = block(x)
        return self.llr_from_head(x, base, self.exits[-1])

    def final_only(self,features,base):
        """Fixed deep reference without evaluating unused intermediate heads."""
        x=torch.nn.functional.silu(self.stem(features))
        for block in self.blocks:
            x=block(x)
        return self.llr_from_head(x,base,self.exits[-1])


def receiver_loss(outputs, bits):
    """Use consistent logit convention: torch BCE logit corresponds to bit 1."""
    return sum(torch.nn.functional.binary_cross_entropy_with_logits(-llr, bits)
               for llr in outputs)/len(outputs)
