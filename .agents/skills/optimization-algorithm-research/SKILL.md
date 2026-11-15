---
name: optimization-algorithm-research
description: Autonomously analyze, derive, prototype, and rigorously evaluate learning-rule improvements in Rust NGBoost and GradientLSS, including natural-gradient conditioning, distributional gradients/Hessians, line search, boosting dynamics, and stability. Use for algorithmic learning quality and convergence research, not code speedups or routine hyperparameter tuning.
---

# Optimization Algorithm Research — Probabilistic Rust Models

## Mission
Independently investigate whether the **learning algorithm itself** can be mathematically improved. You may derive new update rules, investigate alternatives to existing optimization methods, and combine ideas from relevant literature. Do not mistake runtime speedups, extra tuning, or larger models for evidence of a better optimization algorithm. Preserve a stable baseline.

## Repository-first analysis
1. Inspect the actual crate code, public APIs, model assumptions, supported distributions, differentiable losses, parameter transformations, line search, clipping, regularization, RNG, training schedules, and base learners. Read the matching file in `references/`.
2. Trace exactly how objective gradients and second-order/metric quantities are computed, transformed, and consumed in each learning step. Identify which math is analytic, approximate, or numerically stabilized.
3. Establish reproducible baseline training curves, out-of-sample scores, runtime, and numerical diagnostics. Inspect existing research history and known bugs before proposing changes.

## Independent discovery
Investigate observed weaknesses (nonconvergence, sensitivity to parameter scale, poor conditioning, inaccurate gradient/curvature approximation, instability, failure to optimize a proper score, convergence plateaus, high variance across seeds), but also autonomously explore theoretically motivated opportunities not yet exposed by failures. No preset menu of solutions is mandatory.

Candidate directions *only if evidence supports them*: natural-gradient damping and trust regions, Fisher regularization/approximations, step-size or line-search strategies, alternative coordinate/joint parameter updates, proximal or constrained optimization, gradient/Hessian stabilization, curvature-aware shrinkage, adaptive boosting schedules, interaction among parameter learners, or new proper-score-compatible updates.

## Mathematical proposal requirements
For each candidate, specify:
- Distribution, allowed parameter domain, links, differentiability and boundary behavior.
- Optimization objective and whether it is a proper scoring rule for the predictive distribution.
- Baseline update and candidate update (write equations, not only prose).
- Required gradients, Hessians, or Fisher metric and how they are computed.
- Conditions for positivity, symmetry, positive definiteness, and numerical solvability where applicable.
- Computational and memory complexity; convergence argument or explicitly stated absence of one.
- Plausible failure mode, falsifiable prediction, and simpler competing explanation.

Where supported, verify analytic derivatives using centered finite differences or independent autodiff; test Hessian/metric behavior; constrain parameters without covertly changing the model family. For Fisher linear systems, avoid explicit inversion when a stable solve exists and compare conditioning/damping strategies on difficult inputs. These are tests, not prescribed architectural changes.

## Research and experiment cycle
1. **Diagnose:** select an evidence-backed or theoretically justified question; record baseline and alternative hypotheses.
2. **Literature:** research primary papers via authorized sources and Lacuna MCP if connected. Record exact citations and distinguish published ideas from novel conjectures. Do not invent source access.
3. **Derive:** write a design note with equations, assumptions, proof obligations, and expected trade-offs.
4. **Verify:** test analytical identities, correct signs/scaling, domains, and edge cases on small synthetic examples.
5. **Prototype:** implement one isolated experimental algorithm behind a feature flag or branch. Keep reference behavior unchanged and tests passing.
6. **Compare fairly:** identical datasets, features, splits, seeds, training budgets, objectives, and sensible **equal tuning budgets**. Compare convergence per iteration and per wall-clock, plus predictive quality; distinguish whether one method simply got more compute.
7. **Evaluate:** held-out NLL/log score, CRPS when appropriate, calibration, tails, worst-case failures, robustness across distributions/seeds/datasets. Include ordinary and adversarial stress cases, plus uncertainty in score differences.
8. **Decide:** reject, revise, or preserve as an opt-in experiment. Report regressions and conditions of applicability. Select the next question from actual evidence.

## Crucial guards
- A lower training objective is **not** proof of better held-out performance. Conversely faster convergence is not automatically better CRPS.
- Don't optimize only on one adversarial dataset, only on MLB/NBA, or only one distribution. Test broader applicability when claiming a crate-level improvement.
- Do not change default algorithm, public APIs, serialization, score definition, or model capacity without approval.
- Maintain clean chronological validation for sports/time-series data; final test remains untouched during exploration.
- Constrain paid compute, GPU hours, dependencies, and experimental duration to an explicitly approved budget.
- If numerical diagnostics show a correctness bug, escalate to `adversarial-testing` and add a regression before claiming mathematical innovation.
- If improvements rely on a different hyperparameter configuration, coordinate fair searches with `automl-hyperparameter-research` rather than tuning only the candidate.

## Working artifacts
Store under `research/optimization-algorithm/`:
- `research_log.md` — questions, alternative hypotheses, reading, findings, negative results.
- `proposals/` — objective, baseline/candidate equations, derivation and numerical tests.
- `experiments.csv` — commit, seed, dataset/split, scores, hyperparameter budget, time, failures, decision.
- `reports/` — reproducible commands, comparisons and independent verification.

## Report format
**Observed limitation or theoretical question → Baseline mathematics → Candidate mathematics → Verification tests → Controlled comparison → Out-of-sample results → Runtime/convergence trade-off → Decision → Next research question.** Continue autonomously while budget remains; no fabricated results or implied approval to merge.
