"""Paired-exit decision utilities, not a claim of methodological novelty."""
import numpy as np


def outcome_labels(shallow_errors, deep_errors):
    """Truth-derived training labels: 0 unchanged, 1 beneficial, 2 harmful.

    Never call this in a receiver's inference path. Bit error counts are
    reduced to block correctness, matching the BLER objective.
    """
    s = np.asarray(shallow_errors) > 0
    d = np.asarray(deep_errors) > 0
    if s.shape != d.shape:
        raise ValueError("paired outcomes must have identical shapes")
    return np.where(s & ~d, 1, np.where(~s & d, 2, 0))


def choose_continue(probabilities, extra_cost, error_penalty):
    """Two-action conditional Bayes rule for cost + penalty * block_error.

    probabilities[..., :] = [unchanged, beneficial, harmful]. Prediction is
    receiver-observable, but its calibration is an empirical obligation.
    extra_cost must include all incremental path costs in a common unit.
    Common gate cost cancels in this action comparison but MUST be included
    when comparing the method's total cost to gate-free baselines.
    """
    p = np.asarray(probabilities, float)
    if p.shape[-1] != 3 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("invalid outcome probabilities")
    if not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    if np.any(np.asarray(extra_cost) < 0) or error_penalty < 0:
        raise ValueError("costs/penalty must be nonnegative")
    gain = p[..., 1] - p[..., 2]
    return error_penalty*gain > extra_cost


def binary_entropy_from_llr(llr):
    """Mean bit entropy per frame, in nats, computed without overflow."""
    a = np.abs(np.asarray(llr, float))
    q = np.exp(-a)
    return (np.log1p(q) + a*q/(1+q)).mean(axis=-1)


def paired_summary(shallow_errors, deep_errors):
    labels = outcome_labels(shallow_errors, deep_errors)
    return {"frames": labels.size, "shallow_bler": float((np.asarray(shallow_errors)>0).mean()),
            "deep_bler": float((np.asarray(deep_errors)>0).mean()),
            "beneficial_fraction": float((labels==1).mean()),
            "harmful_fraction": float((labels==2).mean()),
            "signed_gain": float((labels==1).mean()-(labels==2).mean())}


def lean_feature_transform(entropy, pilot_residual):
    """Small nonlinear basis; no quantiles, sorting, or extra FFT required."""
    e=np.asarray(entropy)
    r=np.log1p(np.maximum(np.asarray(pilot_residual),0))
    return np.stack([e,e*e,r,e*r],axis=-1)


def lean_observable_features(observation, shallow, cfg):
    """Buffered-frame pilot variation plus soft entropy; no truth/noise oracle."""
    grid=observation.grid
    pilots=grid[...,cfg.pilots%cfg.fft_size]
    scale=np.maximum(np.mean(abs(grid[...,cfg.active%cfg.fft_size])**2,axis=(1,2)),1e-9)
    residual=np.mean(abs(pilots-pilots.mean(axis=1,keepdims=True))**2,axis=(1,2))/scale
    return lean_feature_transform(binary_entropy_from_llr(shallow),residual)
