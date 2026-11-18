# GPU plan: memory along the loop axis (looped reasoners) and along training (gradient noise)

Written 2026-11-11 so that a session on a machine with a GPU can run it from a fresh checkout of this branch. Context:

- `results/looped_filter_fgn/README.md`: the link to Rozwood, *Looped Reasoning as Closed-Loop Filtering* (draft v0).
- `results/forgetting_killtest/README.md`: two EMAs are within about 1% of the optimal filter for long-memory drift.

Run Part A first: it is quick and decides whether the optimizer route stays open. Part B (loop-axis memory), Part C (does the gain change which solution is found?) and Part D (a better filter: Koopman / Kalman against Anderson acceleration) are the looped-reasoner line; C and D bear most directly on the draft paper.

All predictions below are fixed **before** running. Report every one with its outcome, in the style of the draft's appendix C. Use `fsde.hurst_second_diff` and `fsde.hurst_bootstrap` for H. Never use first-difference variograms; they are biased by drift (`docs/phase0.md`).

## Part A: does minibatch gradient noise have long memory? (about 1 GPU-hour)

**Why.** If gradient noise is close to white along training, power-law or fractional gradient filtering has nothing to exploit, and the optimizer route is closed. A paper claiming long memory in SGD noise (arXiv 2105.02062) was withdrawn over a code bug, so the measurement must be careful.

**Setup.**

1. ResNet-18 on CIFAR-10 with SGD, momentum 0, batch 128, **constant** learning rate, so no schedule makes the series non-stationary. Train about 5 epochs (≈ 2,000 steps) and record from step 500.
2. At each recorded step t:
   - compute the minibatch gradient g_t;
   - compute a reference gradient G_t on a fixed, large held-out batch of 4,096 training examples, at the same parameters;
   - take the noise n_t = g_t − G_t.
3. Project n_t onto 32 fixed random unit directions and onto the top 8 Hessian eigenvectors. Get the eigenvectors by Lanczos at step 500, and recompute them every 500 steps. This gives 40 scalar series.
4. For each series, estimate H of the noise as fGn: apply `hurst_second_diff` to its cumulative sum, and report a bootstrap CI.
5. Also report H of the projected reference gradient G_t, i.e. the drift.
6. **Controls:**
   - Repeat with independently resampled minibatches at fixed parameters (expected exactly white, H = 0.5).
   - Repeat with i.i.d. synthetic fGn at H = 0.3 / 0.5 / 0.7 run through the same code, to check that the pipeline can tell them apart.

**Registered predictions.**

- A1. The median H of noise projections over the 40 directions lies in [0.45, 0.55].
- A2. No more than 4 of the 40 directions have a 95% CI entirely above 0.6.
- A3. H of the reference gradient G_t projections is above 0.8, i.e. smooth, persistent drift.

**Decision rule.** If A1 fails because the median is above 0.6, the optimizer route revives. Then test fractional or two-EMA momentum against AdamW at matched tuning budget. Otherwise close it.

## Part B: memory along the loop axis in trained looped reasoners

**Checkpoints and code (all public, as listed in the draft):**

- Mini-ARC: TRM, HRM, EqR, FPRM, SFPRM. Code and checkpoints are at `github.com/arberzela/sfprm`; the Hugging Face repo is `arberzela/sfprm`.
- Sudoku-Extreme:
  - FPRM, Hugging Face `fixed-point-reasoners/fprm`;
  - TRM-attention, two checkpoints: `yagizdevre/trm-sudoku-extreme-att` (which overthinks) and `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` (control).
- InfiLoop weights are Preston's own, from his retraining; retraining from the released recipe takes about 26 h on one GPU. Ask Preston for them before retraining.

**Rollout.** For each model, run 256 puzzles (Sudoku) or all 149 tasks (Mini-ARC) for 16× the trained depth under the model's own update rule, and save per outer step:

- the state h_t and the candidate z_t = f(h_{t−1}), each projected onto (i) the readout's directions (the decoded subspace, as in the draft's §9) and (ii) 64 fixed random directions in the scratch / prefix subspace;
- the decoded answer, whether it is correct, and the relative residual ‖f(h) − h‖ / ‖h‖.

Save as `npz`, one file per model.

**Measurements (CPU is fine once the rollouts exist).**

1. **Innovation memory:** H of the innovations e_t = z_t − h_{t−1}, per puzzle and projection, taken over the *search phase* (before the residual first drops below 10⁻²) and, separately, over the post-solution phase.
2. **Trajectory scaling:** the exponent of E‖h_{t+k} − h_t‖² ∝ k^(2H_traj) over k = 1..32 during search. Report it for solved, lost (correct at the trained depth, wrong later) and never-solved puzzles.
3. **Label-free halting:** does a rolling H_traj or innovation-H window predict a later loss of the answer better than the residual alone? Score by AUROC on the overthinking TRM checkpoint.

**Registered predictions.**

- B1. In at least 3 of the 6 public models, the median search-phase innovation H, in readout directions, differs from 0.5 by more than 0.1. Under the null, §5's white-noise assumption holds.
- B2. Post-solution innovations in held solutions are anti-persistent (H < 0.45) in models whose binding mode is a real flip mode below −1 (FPRM Sudoku; see the draft's §7).
- B3. On the overthinking TRM checkpoint, puzzles that are later lost have a higher search-phase H_traj than puzzles that are held (one-sided Mann–Whitney, p < 0.05). The control checkpoint shows no such split, because it loses no answers.
- B4. The rolling-H halting signal improves AUROC for predicting a lost answer by at least 0.05 over the residual-only signal.
- B5. This needs InfiLoop weights. If the search-phase innovation H for InfiLoop's outer state is above 0.55, replacing its single outer decay at inference with a tuned two-decay filter (an AdEMAMix-style mixture of two EMAs with a learned or tuned mixing weight; tune on 64 held-out puzzles, evaluate on the other 192) raises the share of puzzles solved after the trained depth by at least 2 points. If H ≤ 0.55, it does not.

**Outputs.** Commit `results/looped_memory/` containing the measurement script, the per-model tables, and a README listing every prediction with its outcome. Commit raw rollouts only if they are small (under about 50 MB); otherwise keep them local.

## Part C: does the gain change *which* solution is found? (about one GPU-day)

**Why.** The draft explains *holding*; accuracy is decided by *finding* (its §10). The ODE method says that as γ → 0 the loop tracks the flow ḣ = f(h) − h, so which solution is found is set by the flow's basins. Prior art (Equilibrium Reasoners, the HRM and TRM attractor papers, Anil et al. 2022) already claims that basins decide finding, but nobody has measured, solution by solution, whether *damping changes the basin a puzzle lands in*. If it does not, the ODE theory adds little beyond Euler steps and should be dropped.

**Setup.**

- **Models:** carry-last models first (TRM, HRM and EqR on Mini-ARC; both TRM checkpoints on Sudoku), then FPRM / SFPRM.
- **Runs:** for each puzzle, start from the model's initial state. Run the loop at a constant gain γ ∈ {1, 0.5, 0.25, 0.1} for 16 × trained depth / γ block evaluations, so all gains reach the same flow time.
- **Flow reference:** integrate ḣ = f(h) − h with an adaptive solver (RK45, rtol 1e−5) to the same flow time.
- **Recorded per puzzle and per run:** the decoded answer, whether it is correct, the final residual, and the distance between end states of different runs. Two runs "agree" if their decoded boards match and their end states are within 1% relative distance.
- **Lyapunov check:** along each flow trajectory, record whether the residual ‖f(h) − h‖² decreases monotonically.

**Registered predictions.**

- C1. For the carry-last models on Mini-ARC, the decoded answer differs between γ = 1 and γ = 0.1 for at least 10% of tasks. **Kill criterion:** below 5% means the basins do not depend on the gain, and the ODE theory is dropped from the paper.
- C2. Agreement between the loop's end state and the flow's end state rises monotonically as γ falls (1 → 0.5 → 0.25 → 0.1), and exceeds 90% at γ = 0.1.
- C3. On the overthinking TRM checkpoint, puzzles where the γ = 1 run and the flow disagree are over-represented among the puzzles that carry-last loses between steps 16 and 256 (odds ratio > 2, Fisher exact p < 0.05).
- C4. The residual decreases monotonically along at least 90% of flow trajectories for EqR (trained with damping and noise). The overthinking TRM checkpoint has a lower share than the control checkpoint.

**Outputs.** `results/looped_basins/` with the script, the per-model agreement tables and a README listing every prediction with its outcome.

## Part D: a better filter for the loop (Koopman / Kalman), against strong baselines

**Why.** The draft's §5.3 and §12 propose, untested, a filter "with a precision per channel or a gain that propagates uncertainty like a Kalman filter". InfiLoop's filter changes how many solutions are found (39% → 78% across decays, Fig. 7b). A better filter is the practical improvement the paper lacks.

Koopman theory is valid here: unlike the fSDE work, the loop's state evolves under a deterministic Markov map, and the Koopman spectrum at a fixed point contains the Jacobian eigenvalues the draft measures.

**The bar.** Koopman/DMD extrapolation of fixed-point iterates is closely related to Krylov acceleration (DMD ≈ Arnoldi; Anderson acceleration ≈ GMRES). A Koopman filter must therefore beat **Anderson acceleration** at matched block evaluations, or it is a known accelerator under a new name.

**Prior art (search of 2026-11-18; snippets only, full texts not read).** Inference-time solver swaps on looped reasoners are largely taken:

- Spherical Fixed-Point Reasoners (Zela et al. 2026, arXiv 2610.25646) swap in Anderson and Broyden at inference. Anderson halts earlier with about the same accuracy on Maze.
- Park & Rusch 2026 (arXiv 2610.18174) use Anderson to find the roots of FPRM and HRM and study the local Jacobian spectra.
- TAPS (Wang et al. 2026, arXiv 2609.36653) is a training-free adaptive step size for TRM.
- Solve the Loop (Fein-Ashley & Rashidinejad, ICML 2026) uses an Anderson root-finder.
- Invariant-Measure Reasoners (Inui, Konishi, Kawahara 2026, arXiv 2610.10996) bring Koopman-style ideas to looped reasoners.
- Per-dimension Kalman residual updates exist in the Precision Tracked Transformer (arXiv 2605.18832) and Recurrent Kalman Networks (ICML 2019).
- For a linearised loop, windowed DMD ≈ MPE ≈ Krylov ≈ Anderson (Sidi; Rowley et al. 2009).

Consequences for this part:

- D3 and TAPS are **baselines**, not contributions.
- D-1 and D-2 are replications.
- Novelty rests on D-4 (kernel lifting beats Anderson during search) and D-5 (a per-channel Kalman filter for InfiLoop); both are medium at best.
- Add TAPS as variant D7 if its code is public.
- Cite all of the above.

**Variants** (all applied at inference to existing checkpoints; compute is counted in block evaluations):

| ID | Update rule | Notes |
| --- | --- | --- |
| D0 | carry-last (γ = 1) | baseline |
| D1 | constant damping at 0.9 × the per-puzzle limit | the draft's own rule (§11) |
| D2 | InfiLoop's learned EMA filter at the trained decay | InfiLoop only |
| D3 | **Anderson acceleration**, memory 5, mixing 1 (also 0.5) | the key baseline |
| D4 | windowed linear DMD / minimal-polynomial extrapolation, every 8 steps, in the readout subspace plus the top 32 PCA directions of recent iterates; carry-last on the rest | Koopman with linear observables |
| D5 | **kernel Koopman–Kalman filter** (see below) | the candidate method |
| D6 | **per-channel Kalman gain**: Γ_c = P_c / (P_c + R_c), with R_c the running innovation variance of channel c and P_c propagated as in a local-level model | the draft's §5.3 suggestion |

**D5 in detail.**

1. **Per-position features:** project each position's state onto the readout subspace plus the top k = 32 PCA directions of recent iterates. Lift with Nyström RBF features on m = 128 landmarks, maintained online by k-center selection; reuse `OnlineKCenter` from `pomdp-koopman-control/src/sskf/streaming_sig_kkf.py`, which is generic despite its name.
2. **Operator:** learn the lifted operator z_{t+1} ≈ K z_t online by recursive least squares with forgetting, pooling all positions of the current puzzle. Reuse `OnlineKoopman` from `rkhs-kronic/src/online_koopman.py`.
3. **Kalman fusion:** predict ẑ = K z, observe the lifted candidate φ(f(h)), fuse them with the RLS covariance as the prior and the innovation variance as R (the heteroskedastic pattern of `pomdp-koopman-control/src/sskf/heteroskedastic_kalman_v.py`), and decode back to the projected state with a ridge decoder.
4. **Unmodelled directions:** take the candidate unchanged (carry-last) outside the projected subspace.

**Metrics.**

- Share solved at budgets of 16, 32, 64 and 256 block evaluations.
- Block evaluations needed to settle (relative residual below 10⁻³).
- On the overthinking TRM checkpoint, answers lost between budgets 16 and 256.
- Hold tests at solutions: perturb by 10⁻⁴, iterate 2,000 steps under each filter, and record whether the state returns.

**Registered predictions.**

- D-1 (replication of SFPRM / Park & Rusch). Anderson (D3) cuts the block evaluations to settle by at least 30% against carry-last on held solutions of TRM-control, HRM and EqR, and lowers final accuracy by no more than 1 point.
- D-2 (replication). On FPRM (Sudoku) and SFPRM (Mini-ARC), whose solutions carry-last cannot hold, D3 or D4 settles at least 80% of the solutions the model's own adaptive solver settles, at no more than 50% of its block evaluations.
- D-3. This prediction is deliberately sceptical. On the overthinking TRM checkpoint, no variant among D3–D6 recovers more than half of the answers carry-last loses between budgets 16 and 256: those losses are search failures far from fixed points (§10).
- D-4. This prediction carries the novelty. The kernel Koopman–Kalman filter (D5) beats Anderson (D3) in share solved at a budget of 32 block evaluations on at least 2 of the 5 Mini-ARC models, by at least 2 points. If it fails, the Koopman framing reduces to acceleration and is reported as such.
- D-5. This needs InfiLoop weights. On InfiLoop, the per-channel Kalman gain (D6) raises the share solved after the trained depth by at least 2 points over the scalar learned filter (D2) at the trained decay.
- D-6. This extends the gain-limit theory to matrix gains. For D6, the linearised loop at a fixed point is (I − Γ) + Γ J_f with Γ diagonal. Its spectral radius, estimated by JVP power iteration as in the draft's appendix B, predicts the outcome of the hold test in at least 95% of solutions.

**Stage 2** (only if D-4 or D-5 holds): train InfiLoop with the winning filter inside the loop, about 26 GPU-hours per run, with two seeds, and compare against the released recipe.

**Outputs.** `results/looped_filters/` with the filter implementations, the per-model tables and a README listing every prediction with its outcome.
