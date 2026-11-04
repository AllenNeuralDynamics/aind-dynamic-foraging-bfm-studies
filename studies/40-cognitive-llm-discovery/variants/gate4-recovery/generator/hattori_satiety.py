"""Hattori2019 with satiety: Q-learning with separate rates after reward and no reward and forgetting
of the unchosen side, whose softmax inverse temperature falls as rewards accumulate in a session.
A generating program for gate 4 (study 40, variant gate4-recovery); in neither the seeds nor the
library."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rew": (0.0, 1.0),
    "learn_rate_unrew": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "biasL": (-5.0, 5.0),
    "softmax_inverse_temperature": (0.0, 100.0),   # at the start of the session
    "satiety_rate": (0.0, 0.02),                   # per reward: beta * exp(-rate * rewards so far)
}
N_LATENT = 3
LATENT_NAMES = ("q_left", "q_right", "rewards_so_far")


def init_latents(p, session):
    return jnp.zeros(3)


def predict(latents, p):
    beta = p["softmax_inverse_temperature"] * jnp.exp(-p["satiety_rate"] * latents[2])
    return beta * latents[:2] + jnp.array([p["biasL"], 0.0])


def update(latents, choice, reward, p):
    q = latents[:2]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rew"], p["learn_rate_unrew"])
    learned = q + rate * (reward - q)
    forgotten = (1.0 - p["forget_rate_unchosen"]) * q
    q = jnp.where(chosen > 0, learned, forgotten)
    return jnp.concatenate([q, jnp.array([latents[2] + reward])])
