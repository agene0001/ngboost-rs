---
name: adversarial-testing
description: Autonomously discover, minimize, and reproduce mathematical, numerical, statistical, and Rust API failures in NGBoost or GradientLSS using property-based tests, fuzzing, and synthetic data-generating processes. Use when investigating robustness, unexpected predictive failures, panics, instability, and hidden edge cases.
---

# Adversarial Testing & Fuzzing — Probabilistic Rust Models

## Mission
Independently discover **unknown failure modes** in the actual crate. Do not assume its implementation matches Python NGBoost or XGBoostLSS. A successful cycle creates a reproducible, minimal counterexample, a severity assessment, and a falsifiable explanation; **finding no bug is not evidence of correctness**.

## Start with repository reality
1. Inspect `Cargo.toml`, source, distribution implementations, training entry points, existing tests, feature flags, numerical libraries, RNG, and recorded issues. Read `references/ngboost.md` or `references/gradientlss.md` as appropriate.
2. Identify testable invariants: finite objective for valid parameters, normalized probability/CDF, monotone CDF, parameter constraints, shape consistency, predictable errors for invalid inputs, deterministic reproducibility with fixed seed, and serialization round-trip when supported.
3. Create a baseline run with reproducible commands and record compiler/Rust version, feature flags, dataset/seed, and current failures. Do not change production code while collecting baseline evidence.

## Two separate testing tracks
**A. Correctness and reliability:** generate valid and invalid inputs to expose panics, infinite loops, nonfinite values, overflow, boundary behavior, illegal parameter states, and API contract violations. Invalid inputs should produce documented errors, not undefined model behavior; don't incorrectly demand training success for malformed input.

**B. Statistical and algorithmic stress:** generate *valid* datasets where fitting is challenging and quantify held-out probabilistic quality, convergence, conditional calibration, and tail behavior. Poor fit on a correctly implemented model is a statistical limitation, not automatically a software bug.

## Independent adversarial search
Choose tests based on observed gaps, not a fixed checklist. Candidates include: all-identical predictors; constant target; rare events; class/count imbalance; heavy tails; excess zeros; outliers; extreme scale; tiny/large sample-to-feature ratios; correlated features; almost nonidentifiable distribution parameters; covariate shift; abrupt regime change; sparse and dense inputs; extreme-but-valid parameter settings; and very different target magnitudes. Check finite inputs separately from NaN/Inf and invalid shapes.

Use seeded synthetic generators with known data-generating distributions and independently implemented mathematical oracles. Vary multiple factors while also maintaining one-factor-at-a-time controls. Search for failures in both fitting and prediction. When real-data stress cases are available, use them without leaking labels into the pre-prediction feature set.

## Rust tool use (adapt to the crate)
- Prefer built-in Rust test infrastructure, `proptest` for shrinking, `cargo-fuzz`/libFuzzer for byte-level parsers or bounded API fuzz targets, and deterministic generators for model-fitting tests. Check existing dependencies before adding any. Do not assume these tools are installed.
- Set timeouts, memory ceilings, thread limits, and sample-size bounds. Use `cargo test` before expensive model-fitting fuzz campaigns.
- Make stochastic failures replayable: save seed, generator parameters, config, build flags, input fixture or fixture-generation code, and failure trace.
- Shrink each failure to the smallest dataset/parameter combination that still reproduces it; rerun under release and debug when meaningful.
- Add a regression test for every confirmed implementation failure. Distinguish platform-specific numerical errors from deterministic logic defects.

## Required scientific investigation loop
1. **Identify coverage gaps** by inspecting interfaces, code paths, and prior results.
2. **Generate adversarial hypotheses**: at least one correctness hypothesis and one statistical stress hypothesis when practical.
3. **Construct bounded tests** and record all generation parameters.
4. **Run and triage**: separate crash, nonfinite output, invalid probability, gradient error, convergence issue, calibration failure, and legitimate model misspecification.
5. **Minimize** a discovered counterexample; verify against a mathematical oracle, reference implementation, or expected invariant when possible.
6. **Explain** likely mechanism, alternative explanations, and test limitations. Never call a probabilistic score worse than baseline a correctness bug without evidence.
7. **Hand off** reproducible cases to `optimization-algorithm-research` only after establishing what failed; use `automl-hyperparameter-research` for configuration sensitivity, not as a substitute for fixing faulty math.
8. **Re-run independent validation** after any fix. Preserve the original failing case but do not optimize a proposed algorithm exclusively on the newly discovered stress data.

## Evaluation and boundaries
Report prevalence, impact, and reproducibility. For statistical tests compare held-out NLL/log score, CRPS when defined, calibration/coverage or discrete randomized PIT, error rate, and convergence. Include ordinary benchmarks beside pathological ones. Use multiple seeds, and label exploratory results. Never tune against final held-out data. Avoid unbounded `cargo fuzz`, paid resources, or production mutations without approval.

## Records
Store under `research/adversarial-testing/` (or an existing research root):
- `findings.md` — hypotheses, severity, minimal reproduction, root-cause confidence, regression status.
- `cases/` — deterministic minimal datasets or fixture generators; preserve privacy and license constraints.
- `runs.csv` — timestamp, commit, crate/version, seed, generator, test type, runtime, symptom, reproducibility, status.
- `reports/` — commands, environmental details, comparisons, and follow-ups.

## Completion protocol
Produce: **Discovery → Minimal reproduction → Expected invariant / statistical baseline → Actual behavior → Root-cause hypotheses → Evidence → Severity → Regression test → Follow-up experiment**. Continue with the next independent investigation within the authorized budget. Do not silently alter production defaults or public APIs.
