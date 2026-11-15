# NGBoost adapter — adversarial testing

Confirm these components exist before testing them: scoring rule; natural-gradient metric; distribution parameter transform; base learner; boosting loop; prediction interface.

- Test Fisher information symmetry, finite entries, conditioning, and appropriate positive definiteness on valid identifiable parameter regions. Near singularity is often expected; determine whether the solver gracefully handles it.
- Compare score gradients and metric computations to independent numerical references at ordinary and boundary-near parameters.
- Generate rare-count, zero-heavy, heavy-tail, extreme-location/scale, and highly correlated-feature datasets matched to supported families.
- Check learning-rate / line-search / clipping interactions and stable behavior as data scale changes.
- Verify empirical CDF normalization/monotonicity and valid quantiles if exposed; verify predictive samples and seeded replay if supported.
- For count targets, use randomized PIT rather than naive continuous PIT, and score exact event probabilities when possible.
- Preserve minimum failure fixtures with parameter vectors, samples, seeds, and reproduction commands.
