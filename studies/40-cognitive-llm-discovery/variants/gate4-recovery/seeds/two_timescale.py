"""Reward history at two timescales: a fast and a slow reward trace for each side, learned on the
chosen side, and a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "rate_fast": (0.2, 1.0),         # the fast trace's learning rate
    "rate_slow": (0.0, 0.2),         # the slow trace's learning rate
    "weight_fast": (0.0, 20.0),      # logits per unit of the fast trace
    "weight_slow": (0.0, 20.0),      # logits per unit of the slow trace
    "bias_right": (-3.0, 3.0),
}
N_LATENT = 4
LATENT_NAMES = ("fast_left", "fast_right", "slow_left", "slow_right")


def init_latents(p, session):
    return jnp.zeros(4)


def predict(latents, p):
    return p["weight_fast"] * latents[0:2] + p["weight_slow"] * latents[2:4] + jnp.array([0.0, p["bias_right"]])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    fast = latents[0:2] + chosen * p["rate_fast"] * (reward - latents[0:2])
    slow = latents[2:4] + chosen * p["rate_slow"] * (reward - latents[2:4])
    return jnp.concatenate([fast, slow])
