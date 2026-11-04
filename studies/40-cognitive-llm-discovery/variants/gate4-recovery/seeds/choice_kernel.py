"""Choice kernel: a decaying trace of past choices pulls toward recently chosen sides; no learning
from reward."""
import jax.numpy as jnp

PARAMS = {
    "kernel_rate": (0.0, 1.0),       # how fast the kernel follows the last choices
    "kernel_weight": (-5.0, 5.0),    # logits toward the side the kernel favours
    "bias_right": (-3.0, 3.0),
}
N_LATENT = 1
LATENT_NAMES = ("kernel",)           # +1 after many right choices, -1 after many left ones


def init_latents(p, session):
    return jnp.zeros(1)


def predict(latents, p):
    return jnp.array([0.0, p["kernel_weight"] * latents[0] + p["bias_right"]])


def update(latents, choice, reward, p):
    return latents + p["kernel_rate"] * (2.0 * choice - 1.0 - latents)
