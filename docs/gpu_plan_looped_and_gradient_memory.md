# GPU plan: memory along the loop axis (looped reasoners) and along training (gradient noise)

Written 2026-11-11 so that a session on a machine with a GPU can run it from a fresh checkout of this branch. Context:

- `results/looped_filter_fgn/README.md`: the link to Rozwood, *Looped Reasoning as Closed-Loop Filtering* (draft v0).
- `results/forgetting_killtest/README.md`: two EMAs are within about 1% of the optimal filter for long-memory drift.

Run Part A first: it is quick and decides whether the optimizer route stays open. Part B is the main line.

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
