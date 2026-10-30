# Stability-aware symbolic distillation (Pendulum-v1)
Run in order (1 CPU: ~1.5 h total):
1. `python3 region.py`            -> LQR Lyapunov candidate, region D, alpha (results/region.json)
2. `python3 test_sim.py`          -> checks the simulator matches Gymnasium
3. `for s in 0 1 2 3 4; do python3 train_teacher.py $s; done`   -> SAC teachers
4. `python3 experiment.py`        -> DAgger students + hyper-parameter sweep on VALIDATION states
5. `python3 test_stage.py`        -> applies selection rules, runs 22 conditions + RoA on TEST states
6. `python3 analysis.py && python3 tables.py`  -> figures, statistics, LaTeX tables
Versions: Python 3.12, gymnasium 1.2.x, stable-baselines3 2.x, torch 2.x, scikit-learn 1.x.
