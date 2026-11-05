"""One reward trace for each side plus a choice kernel: the generating program without its slow
trace. The pre-check scores it, and the generating program must beat it by 0.4 (notes.md); it is
never shown to Claude."""
import jax
import jax.numpy as jnp

PARAMS = {
    "rate": (0.0, 1.0),              # the trace's learning rate, chosen side only
    "weight": (0.0, 20.0),           # logits per unit of the trace
    "kernel_rate": (0.0, 1.0),
    "kernel_weight": (-5.0, 5.0),
    "bias_right": (-3.0, 3.0),
}
N_LATENT = 3
LATENT_NAMES = ("value_left", "value_right", "kernel")


def init_latents(p, session):
    return jnp.zeros(3)


def predict(latents, p):
    return p["weight"] * latents[0:2] + jnp.array([0.0, p["kernel_weight"] * latents[2] + p["bias_right"]])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    value = latents[0:2] + chosen * p["rate"] * (reward - latents[0:2])
    kernel = latents[2] + p["kernel_rate"] * (2.0 * choice - 1.0 - latents[2])
    return jnp.concatenate([value, jnp.array([kernel])])
