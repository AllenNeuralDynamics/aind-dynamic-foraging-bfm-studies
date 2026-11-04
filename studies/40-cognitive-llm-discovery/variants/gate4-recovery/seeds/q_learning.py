"""Q-learning: one value for each side, learned from reward on the chosen side, and a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate": (0.0, 1.0),        # how far the chosen side's value moves toward the reward
    "beta": (0.0, 20.0),             # logits per unit of value
    "bias_right": (-3.0, 3.0),
}
N_LATENT = 2
LATENT_NAMES = ("q_left", "q_right")


def init_latents(p, session):
    return jnp.zeros(2)


def predict(latents, p):
    return p["beta"] * latents + jnp.array([0.0, p["bias_right"]])


def update(latents, choice, reward, p):
    chosen = jax.nn.one_hot(choice, 2)
    return latents + chosen * p["learn_rate"] * (reward - latents)
