# Certificate-Guided Symbolic Distillation of Deep RL Controllers (Pendulum-v1)

Code, trained models and results for the paper *Certificate-Guided Symbolic Distillation of Deep
Reinforcement Learning Controllers with Formally Verified Lyapunov Decrease* (G. F. Tano-Boamah).

A soft actor-critic (SAC) teacher is distilled with DAgger into a sparse symbolic controller (ten odd
monomials of cos θ, sin θ, θ̇). A sound interval-arithmetic verifier proves the Lyapunov decrease
condition V(x⁺) ≤ (1−α)V(x) on a region D near upright, and its counterexamples are fed back into
training (certificate-guided distillation).

**Headline results (5 teacher seeds):** certificate-guided students are formally certified on
96.2 ± 1.6% of D (saturated LQR reference 99.6%), versus ≈0% for plain distillation and 14.9 ± 20.0%
for an absolute Lyapunov penalty; robust success over 22 test conditions is 84.1% versus 68.0% for
plain distillation (Wilcoxon p = 4.4 × 10⁻⁷), at a cost in return (−187.2 versus −157.0).

## Requirements
Python 3.12, gymnasium 1.2.x, stable-baselines3 2.x, torch 2.x, scikit-learn 1.x, numpy, scipy,
matplotlib. All runs use one CPU core.

## Reproduction

### Part 1: original pipeline (teachers, baselines, test protocol), about 1.5 h
| Step | Command | Output |
|---|---|---|
| 1 | `python3 region.py` | LQR Lyapunov candidate P, region D, α (`results/region.json`) |
| 2 | `python3 test_sim.py` | checks the simulator matches Gymnasium (max return error ≈ 5×10⁻⁵) |
| 3 | `for s in 0 1 2 3 4; do python3 train_teacher.py $s; done` | five SAC teachers (`models/`) |
| 4 | `python3 experiment.py` | DAgger students and hyper-parameter sweeps on VALIDATION states |
| 5 | `python3 test_stage.py` | selection rules; 22 conditions and region of attraction on TEST states |
| 6 | `python3 analysis.py && python3 tables.py` | original figures and tables |

### Part 2: revision (formal certification and certificate-guided distillation), about 15 min
| Step | Command | Output |
|---|---|---|
| 7 | `python3 certify_baselines.py` | certifies LQR, plain and stability-aware students (Table 2) |
| 8 | `bash run_sweep.sh` | certificate-guided distillation, λ_r ∈ {1, 3, 10, 30, 100} × 5 seeds (caches DAgger data in `results/dagger_s*.npz`) |
| 9 | `python3 eval_cert.py` | applies the λ_r rule on validation data, evaluates the selected students on the TEST protocol (`results/test2.json`) |
| 10 | `python3 analysis2.py` | tables, statistics, gains, linearised tests, figures (`results/report2.json`, `figs2/`) |
| 11 | `python3 ball_bound.py` | verification timing and the excluded-ball invariance bound |
| 12 | `python3 build_tables.py` | LaTeX tables for the manuscript |

## Main files
| File | Purpose |
|---|---|
| `common.py`, `sim.py` | exact Pendulum-v1 dynamics, LQR, feature library, vectorised simulator |
| `distill.py` | DAgger, sparse symbolic fitting, absolute Lyapunov penalty, tree baseline |
| `certify2.py` | **sound verifier**: natural and mean-value interval forms, outward rounding, branch-and-bound |
| `certify.py` | first-version verifier (natural form only); also provides the linearised test `lmi_ok` |
| `cegis.py` | **certificate-guided distillation**: ratio penalty plus counterexample feedback |
| `results/` | all selection logs, test results and certification outputs |

## Notes
- Settings are chosen only on validation starts (seed 777); results come only from test starts (seed 2026).
- Certificates refer to the nominal model and exclude a ball with V < 10⁻⁴c, handled by the linearised test.
- SAC training is not bit-for-bit deterministic across machines; re-trained teachers may differ slightly.
