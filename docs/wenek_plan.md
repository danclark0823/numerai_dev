# dc_wenek: frozen first experiment

Recorded 2026-10-13 before the first Wenek run. Existing production models,
configuration, predictions, and hosted packages are outside this experiment.

Hypothesis: a low-rank pairwise interaction model trained on strictly historical
out-of-fold ridge errors adds useful signal beyond ridge. Use the existing
streamed ridge solver, alpha 10; eight interaction factors; eight Adam epochs;
learning rate 0.003; L2 penalty 0.01. Feature selection is a seeded, label-free
sample of the same 128 medium features used in the previous experiment.

Train on 1,024 sampled rows per eligible era, every fourth era. Test on every
resolved row in eras 850–879, 1000–1029, and 1133–1162. Inner expanding windows
use 40 sampled eras of warm-up and 32-era blocks; both inner and outer fits have
a strict 13-calendar-era embargo. The 52-era historical holdout and its embargo
are excluded. The development windows have been examined before; they are not
fresh confirmation data. Newly resolved, unexamined eras are needed before a
final promotion decision, and will not be substituted with the old holdout.

The interaction target is **raw target minus raw out-of-fold ridge prediction**.
The interaction-only model retains an intercept but has no trainable linear
terms in its standardized feature coordinates. Test ridge alone, a normal FM on
the original target, the interaction model alone, and fixed 10%/25% interaction
blends of per-era Gaussianized components. Include a normal FM trained on the
same residual-eligible rows as a control for excluded warm-up data. A zero-weight
blend must reproduce ridge exactly. No blending weight search is permitted.

The historical ridge/ExtraTrees/LightGBM teacher ensemble is a secondary proxy
reference only. Fit it on each outer training set, never evaluation labels.
Existing production artifacts are not honest historical baselines on these eras.

Report per-window and pooled CORR, MMC where available, BMC, ridge and teacher
contribution, full pairwise prediction correlation, worst-window means, score
variability, and positive-era rates. Report paired 13-era moving-block bootstrap
intervals **separately within each contiguous window**. Thirty eras provide only
about 2.3 blocks; intervals are exploratory. Do not pool CORR and MMC from
different eras into an aggregate payout figure. Store predictions and per-era
scores so every comparison is auditable.

## Predefined scaling decision

Repeat with 2,048 training rows per era (all other settings fixed) only if at
least one blend improves CORR versus ridge in two of three windows and has a
nonnegative MMC difference in the available MMC window, and the interaction
component has positive contribution beyond ridge in at least two windows.
This is a screening rule, not a statistical claim or deployment approval.
If it fails, record the negative result and stop scaling; do not tune against
the final historical window until a configuration passes by chance.

## Verification and delivery

Test chronology, future-label isolation, exact residual construction, synthetic
interaction recovery, zero linear coefficients, zero-weight parity, sampling
alignment, serialization, numerical-batch invariance, full-era ranking, and
submission validity. Run the complete existing tests and lint. Verify fitted
bundles reproduce saved full-era predictions after reloading. Hash the four
production models/configuration/hosted bundles before and after the run.

Save models, feature order, preprocessing, training boundaries, source hashes,
configuration, metrics, and runtime/memory measurements in separate artifacts.
macOS peak RSS understates physical footprint under compression and is labeled
accordingly. A deployment candidate, full-data refit, and hosted package are
conditional on convincing validation plus available unseen data; no automatic
upload or changes to the current live lineup are part of this experiment.
