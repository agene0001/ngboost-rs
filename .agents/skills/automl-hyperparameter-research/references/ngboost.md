# NGBoost adapter — AutoML research

Enumerate implemented learning-rate, boosting-round, base learner, score, distribution, regularization, and natural-gradient-related controls. Some may be categorical **model choices** rather than ordinary numeric hyperparameters; record that distinction.

If the crate uses a per-parameter Rust `tpe` optimizer, investigate interaction sensitivity and compare with multivariate/conditional strategies where justified. Do not assume a newer algorithm dominates.

Benchmark with identical supported distributions, splits, objective, training budgets, and tuning limits. Record failed/nonconverged configurations instead of dropping them silently. Test whether partial boosting trajectories are reliable predictors of completed-trial performance before pruning.
