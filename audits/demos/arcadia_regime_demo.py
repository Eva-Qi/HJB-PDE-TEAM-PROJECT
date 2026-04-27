# ARCADIA CONTRIBUTION — pending review
# WILL NOT RUN: imports from extensions.regime which has no implementation
import numpy as np

from extensions.regime import fit_hmm, regime_aware_params
from shared.params import DEFAULT_PARAMS

# fake returns: first half calm, second half more volatile
np.random.seed(42)
r1 = 0.005 * np.random.randn(300)
r2 = 0.02 * np.random.randn(300)
returns = np.concatenate([r1, r2])

regimes, states = fit_hmm(returns)

print("Regimes:")
for reg in regimes:
    print(reg)

print("\nFirst 20 states:")
print(states[:20])

print("\nLast 20 states:")
print(states[-20:])

print("\nRegime-adjusted params:")
for reg in regimes:
    p = regime_aware_params(DEFAULT_PARAMS, reg)
    print(reg.label, "sigma =", p.sigma, "gamma =", p.gamma, "eta =", p.eta)
