import numpy as np

# Baseline A: Always do nothing (flat position)
def baseline_flat(state):
    return 0  # action 0 = do nothing


# Baseline B: Simple momentum
# If btc_norm > 0 → buy YES; else buy NO
def baseline_momentum(state):
    btc_norm = state[0]
    if btc_norm > 0:
        return 1  # buy YES
    else:
        return 3  # buy NO


# Baseline C: Random agent
def baseline_random(state, action_space):
    return action_space.sample()
