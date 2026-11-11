"""Template: a side bias and a tendency to repeat the last choice; no learning from reward."""
# EVOLVE-BLOCK-START
"""Every latent follows one delta rule, x <- x + step * (target - x):
  - chosen side's value moves toward the reward (rate depends on reward / no reward),
  - unchosen side's value decays toward 0 (forgetting),
  - fast and slow choice traces move toward the side just chosen (+0.5 right, -0.5 left).
The right-minus-left logit is a weighted sum of the latents plus a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),     # value step after a reward
    "learn_rate_unrewarded": (0.0, 1.0),   # value step after no reward
    "unchosen_forgetting": (0.0, 1.0),     # decay of the unchosen side's value toward 0
    "beta": (0.0, 20.0),                   # inverse temperature on the value difference
    "fast_trace_step": (0.2, 1.0),         # step size of fast choice trace (perseveration)
    "fast_trace_weight": (-3.0, 5.0),      # pull of fast trace, in logits
    "slow_trace_step": (0.0, 0.3),         # step size of slow choice trace (habit)
    "slow_trace_weight": (-3.0, 3.0),      # pull of slow trace, in logits
    "right_bias": (-3.0, 3.0),             # fixed preference for right, in logits
}
N_LATENT = 4
LATENT_NAMES = ("q_left", "q_right", "fast_trace_signed", "slow_trace_signed")


def init_latents(p, session):
    return jnp.zeros(4)


def predict(latents, p):
    # Signals: q_right - q_left, fast trace, slow trace, constant 1 (for the bias).
    signals = jnp.concatenate([jnp.diff(latents[:2]), latents[2:], jnp.ones(1)])
    weights = jnp.array([p["beta"], p["fast_trace_weight"],
                         p["slow_trace_weight"], p["right_bias"]])
    return jnp.stack([0.0, jnp.dot(weights, signals)])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    side = jnp.diff(chosen)  # +1 right, -1 left
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    step = jnp.concatenate([jnp.where(chosen, rate, p["unchosen_forgetting"]),
                            jnp.array([p["fast_trace_step"], p["slow_trace_step"]])])
    target = jnp.concatenate([chosen * reward, side, side])
    return latents + step * (target - latents)
# EVOLVE-BLOCK-END