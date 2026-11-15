---
name: automl-hyperparameter-research
description: Autonomously audit and improve hyperparameter search efficiency, search-space design, pruning, and reproducible model selection for Rust NGBoost and GradientLSS; benchmark Rust TPE against alternatives under equal compute budgets. Use when improving model selection or the search algorithm itself, not to conflate tuning gains with new learning algorithms.
---

# AutoML & Hyperparameter Research — Probabilistic Rust Models

## Mission
Independently improve **how the crate searches for configurations** and how efficiently it selects generalizable models. Research the tuner itself when justified. Separate (A) gains from tuning a model and (B) gains from developing a better search algorithm. Do not assume TPE or Optuna-inspired settings are optimal for this application.

## First inspect the actual infrastructure
1. Read `Cargo.toml`, tuning modules, search spaces, scoring/evaluation code, data split routines, artifact storage, trial scheduler and concurrency model. Determine whether the Rust `tpe` crate, other samplers, or custom implementations are present. Do not assume they are.
2. Document sampling transforms (log, categorical, integer), bounds, constraints, conditional parameters, warm-up, candidate sampling, treatment of failed/pending trials, pruning and repeated seeds.
3. Trace objective values: distributional NLL/CRPS definitions, direction of optimization, handling of nonfinite results, normalization across datasets, and train/validation/test separation.
4. Snapshot baseline search traces and reproducible trials (configuration, seed, wall-time, metrics). Run a random-search control; without this the quality of TPE is difficult to assess.

## Autonomous research directions
Choose based on diagnostics. Potential hypotheses include independent vs multivariate TPE, conditional/hierarchical search spaces, exploration-exploitation balance, density-estimator priors, startup strategy, robust handling of duplicate trials and pending parallel jobs, cost-aware acquisition, multi-fidelity schedules, pruning, warm starts, model-based search alternatives, sensitivity/importance analysis, or automatic constraints. These are examples, not requirements.

Investigate whether the current search budget and search-space design—not the sampler—are the dominant bottleneck. Check whether hyperparameter interactions actually matter. Avoid expanding the search space purely for novelty.

## Two-tier evaluations (mandatory distinction)
**Tier 1 — Tuner evaluation:** On a frozen model, dataset, objective, and search space, compare samplers by best *validation* score reached vs (i) wall-clock/compute budget and (ii) completed trials. Repeat independent tuning runs with different sampler seeds; compare variability and failure rates. Include random search, current TPE, and a justified challenger. Respect feasible fair-parallelism rules.

**Tier 2 — Generalization evaluation:** Select configurations without looking at final test results, retrain under comparable budgets, and evaluate on a separate chronological validation period or untouched test set **once model selection is complete**. Report whether apparent tuner gains transfer to genuinely unseen data.

Report tuner efficiency and trained-model quality as different outcomes; winning more often on the reused validation set can indicate search-induced overfitting.

## Autonomous optimization cycle
1. **Measure:** locate search inefficiencies, parameter interactions, poorly designed ranges, wasted trials, unstable objective estimates, premature pruning, or parallel collisions.
2. **Hypothesize:** formulate competing explanations and expected gain per unit computation.
3. **Research:** consult primary methods literature or connected Lacuna MCP when relevant, and trace ideas to original algorithms. Do not claim unverified novelty.
4. **Prototype:** implement the smallest isolated sampler/search change; preserve a reference tuner. Record exact algorithm choices and RNG seeds.
5. **Benchmark:** compare paired search budgets, a common problem suite, repeated sampler seeds, time-vs-quality curves, stability, and overhead. Use representative NGBoost and GradientLSS objectives, including fast synthetic benchmarks to test algorithm behavior.
6. **Audit:** verify trial accounting, duplicate-trial handling, deterministic replay, correct conditional sampling, invalid configurations, and correctness of reported best score.
7. **Decide:** accept only repeatable improvements under predeclared evaluation criteria; document regressions, limitations, and next hypotheses.

## Pruning and multi-fidelity rules
- Validate that short runs or downsampled datasets predict full-budget ranks before enabling aggressive pruning.
- Record how reductions in rounds/epochs/dataset size affect optimization difficulty and distributional calibration.
- Never compare a pruned candidate and an unpruned baseline under different effective compute without reporting that difference.
- Treat parallel workers and asynchronous pending trials explicitly; prevent duplicated expensive evaluations when feasible.

## Practical guardrails
- If current TPE is already near random/alternative optimizers' best performance within budget, report diminishing returns rather than claiming a better sampler is needed.
- Never adjust the score calculation, distribution, feature set, validation windows, or model architecture to make a sampler appear better.
- Do not tune against untouched final test data; repeated leaderboard evaluation creates selection bias.
- No heavy cluster/cloud runs, paid APIs, massive dependency upgrades, production default changes, or breaking API changes without approval.
- Coordinate with `optimization-algorithm-research`: tune baseline and challenger fairly, but keep learning-rule discovery conceptually separate from tuner discovery.

## Research artifacts
Store under `research/automl-hyperparameter/`:
- `search_space.md` — parameters, transforms, constraints, conditional logic, and rationale.
- `trials.csv` — study, sampler, tuning seed, model seed, config, status, validation score, duration, resource use.
- `study_comparisons.md` — curves, uncertainty across runs, cost budget, and selection rules.
- `research_log.md` — hypotheses, papers, accepted/rejected ideas, next questions.

## Report format
**Bottleneck → Existing sampler/search-space behavior → Competing hypotheses → Controlled tuner benchmark → Generalization check → Compute cost → Decision → Next autonomous experiment.** If tooling is unavailable, provide a reproducible proposed experiment rather than pretending to have run it.
