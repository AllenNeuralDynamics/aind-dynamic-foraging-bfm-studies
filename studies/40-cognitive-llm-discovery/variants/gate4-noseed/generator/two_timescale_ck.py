"""Reward history at two timescales plus a choice kernel: a fast and a slow reward trace for each
side, a choice kernel and a side bias. The hand-set generating program of gate 4 (study 40, variant
gate4-recovery); not shown to Claude."""
import jax
import jax.numpy as jnp

PARAMS = {
    "rate_fast": (0.2, 1.0),         # the fast trace's learning rate, chosen side only
    "rate_slow": (0.0, 0.2),         # the slow trace's learning rate, chosen side only
    "weight_fast": (0.0, 20.0),      # logits per unit of the fast trace
    "weight_slow": (0.0, 20.0),      # logits per unit of the slow trace
    "kernel_rate": (0.0, 1.0),       # how fast the kernel follows the last choices
    "kernel_weight": (-5.0, 5.0),    # logits toward the side the kernel favours
    "bias_right": (-3.0, 3.0),
}
N_LATENT = 5
LATENT_NAMES = ("fast_left", "fast_right", "slow_left", "slow_right", "kernel")   # kernel: +1 right, -1 left


def init_latents(p, session):
    return jnp.zeros(5)


def predict(latents, p):
    value = p["weight_fast"] * latents[0:2] + p["weight_slow"] * latents[2:4]
    return value + jnp.array([0.0, p["kernel_weight"] * latents[4] + p["bias_right"]])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    fast = latents[0:2] + chosen * p["rate_fast"] * (reward - latents[0:2])
    slow = latents[2:4] + chosen * p["rate_slow"] * (reward - latents[2:4])
    kernel = latents[4] + p["kernel_rate"] * (2.0 * choice - 1.0 - latents[4])
    return jnp.concatenate([fast, slow, jnp.array([kernel])])
