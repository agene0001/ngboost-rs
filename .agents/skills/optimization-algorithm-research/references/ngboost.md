# NGBoost adapter — optimization algorithm research

Start from the implemented natural-gradient formula and metric. For a score gradient g and information metric F, investigate the actual solve `F * delta = g`, parameter links, scale factors and direction of the update. Do not assume the same convention as Python NGBoost.

Potential research prompts (not mandatory): stable solves vs explicit matrix inversion; damping/trust-region strategies near singular F; comparisons of empirical, analytic, and approximated Fisher metrics; score-specific metric compatibility; coordinate coupling among distribution parameters; bounded step length; objective descent and convergence; base-learner residual fit and metric preconditioning.

Required experiment: compare baseline and candidate on a supported distribution with independently checked derivatives, known synthetic data-generating process, ordinary and near-degenerate parameter regimes, and at least one further distribution if claiming generality. Report NLL/CRPS, calibration, training stability, and cost separately.
