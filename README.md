# Coded OFDM receiver admission: code and simulation data

Version 0.1.0. Creators: Pengliang Tian (ORCID 0009-0003-5941-9797) and Changjiang Zhang (ORCID 0009-0002-9936-1377), Century College, Beijing University of Posts and Telecommunications, China.

This research archive implements pre-neural admission into a nine-path conventional/neural coded OFDM receiver bank under time-domain burst interference and channel mismatch. It supplies the frozen source, six receiver checkpoints, fitting and validation samples, paired final outcomes, transfer and delay-alignment controls, and complete CPU/MPS timing measurements. All communications data are simulated. The associated manuscript remains private during author review; this release is code and data, not an accepted article or a manuscript preprint.

Download `coded-ofdm-code-data-v1.zip` and its SHA256 manifest from the repository release. Extract into a clean folder. The browsable GitHub source tree contains text files; the release ZIP also contains numerical arrays and binary weights.

## Reproduce stored evidence

Use Python 3.12. Recorded delivery versions are pinned in `requirements-replay.txt`. Packages must be obtained for your platform. No third-party wheelhouse is included.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-replay.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/replay_delivery.py --output replay-new.json
```

Replay verifies every frozen source/checkpoint/selector hash, recomputes main, transfer, alignment and timing aggregates, checks LDPC round-trip behavior, loads six models on CPU and exercises nine soft-bit paths and complete admission. This verifies stored outcomes plus a small receiver smoke check; it is not an independent full-data regeneration. The original full package passed 37 numerical tests. Public-package extraction/replay results are supplied separately after verification.

The scientific freeze is `runs/20261023-admission-study-v2/FROZEN.json`. Main expected frame energy is 271.74748120364274 after deterministic pilot-prefix correction. Earlier exploratory experiments are excluded. The final study has 72 equally weighted conditions, 2,048 frames per condition and three fitted systems. Main arrays have 147,456 rows and nine receiver columns per seed. `NAMES` in `src/adaptive_ofdm/admission.py` declares the column order. `choices.npz` contains branch indices; outcomes are 0/1 block errors. Fitting/validation arrays are separate from final evaluation.

```sh
.venv/bin/python scripts/summarize_admission_final.py --study runs/20261023-admission-study-v2 --output runs/replayed-statistics
.venv/bin/python scripts/plot_admission_paper.py --study runs/20261023-admission-study-v2 --statistics runs/replayed-statistics --timings runs/20261023-admission-timing-v2 --alignment runs/20261023-alignment-control-v1 --output figures/replayed
.venv/bin/python scripts/regenerate_final.py --device cpu --output runs/my-final-rerun
```

All output folders/files must be new. Full evaluation may take substantial time. Use `--device mps` for the original neural backend on a compatible Mac. CPU and MPS floating-point outcomes and timing need not be identical. Recorded timings used Apple M4, 10 CPU cores, 16 GiB memory, macOS 27.0.1 and PyTorch thread setting 4; this setting does not cap every BLAS library. The original frozen runner is unchanged and hardcodes MPS; the portable regeneration helper reuses its functions.

The public plotting utility omits private-distribution figure QA helpers whose redistribution terms were not verified. It retains the numerical plotting code and exports PDF/SVG/PNG/TIFF from supplied outcomes. Arial is requested with DejaVu Sans fallback; typography can differ across hosts. The public utility is not part of the scientific source freeze. Numerical source CSVs are included; manuscript files and private reviews are excluded.

For retraining, run `scripts/train_receiver.py` with each of the six `configs/final-*.json` configurations, an explicitly new output directory and `--device cpu`. The original batch driver names canonical checkpoint folders and must not overwrite supplied results. A new training study requires a separate project copy and a new development/selection freeze. `scripts/evaluate_alignment_control.py` records an untuned post-main-freeze diagnostic; it is not the primary test. The CPU timing directory includes its executed profiler snapshot; the live profiler later gained an explicit device option.

## What the evidence supports

Fixed-window admission has aggregate BLER 0.2419004 versus 0.2838677 for fixed deep reception. Classical-only admission already achieves 0.2453749. Complete CPU means are 2.46215 ms for full admission, 2.44045 ms for classical-only admission, 2.39324 ms for independent shallow reception and 2.73784 ms for fixed deep reception. Most of the apparent gain comes from conventional model selection. Admission has reliability penalties on nominal and excluded channel profiles.

A separate known-delay-compensated control has BLER 0.1255968/0.1235894/0.1230288 for admission/shallow/deep reception. The main advantage does not establish superiority after synchronization. Conditional confidence intervals concern paired frames given three fitted systems. The archive establishes no RF hardware result, real-time deadline, energy saving or arbitrary-shift guarantee. Extensive OpenAI Codex assistance was used for code, analysis and drafting; final human scientific review remains necessary.

## Reuse

Original project code is offered under MIT (`LICENSE`); generated simulation data, model weights and source CSVs under CC BY 4.0 (`DATA_LICENSE.md`). Dependencies retain their own licenses and are not relicensed or redistributed here. No grant funding or competing interests were reported by the authors. Author CRediT roles are pending and are not inferred from repository ownership.
