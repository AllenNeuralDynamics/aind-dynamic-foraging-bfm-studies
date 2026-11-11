# EVOLVE-BLOCK-START
"""Signed-value model. Every latent is a signed (right +, left -) quantity already
scaled by its decision weight, so the right-vs-left logit is their plain sum plus a
side bias. Values: dual-rate Rescorla-Wagner on the chosen side, forgetting toward 0
on the unchosen side. Habits: fast and slow signed choice traces; the fast trace's
weight changes with training experience (log number of sessions)."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),
    "learn_rate_unrewarded": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "value_inverse_temperature": (0.0, 20.0),
    "fast_habit_step_size": (0.2, 1.0),
    "fast_habit_weight": (-6.0, 10.0),
    "slow_habit_step_size": (0.0, 0.3),
    "slow_habit_weight": (-6.0, 6.0),
    "right_bias": (-3.0, 3.0),
    "habit_experience_gain": (-3.0, 3.0),  # change of fast-habit weight per log(sessions)
}
N_LATENT = 5
LATENT_NAMES = ("minus_q_left_scaled", "q_right_scaled", "fast_habit_weighted",
                "slow_habit_weighted", "fast_habit_weight_this_session")


def init_latents(p, session):
    # Fast-habit weight for this session: grows/shrinks with log(1 + sessions), centred ~20.
    fast_weight = p["fast_habit_weight"] + p["habit_experience_gain"] * (
        jnp.log1p(session["index"]) - 3.0)
    return jnp.array([0.0, 0.0, 0.0, 0.0, fast_weight])


def predict(latents, p):
    # All learned latents are signed and weighted: the logit is their sum.
    return jnp.array([0.0, jnp.sum(latents[:4]) + p["right_bias"]])


def update(latents, choice, reward, p):
    # One delta rule: each latent moves toward (signed choice x its weight) at its own rate.
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    rates = jnp.concatenate([
        jnp.where(chosen, rate, p["forget_rate_unchosen"]),
        jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"], 0.0])])
    targets = (choice - 0.5) * jnp.concatenate([
        p["value_inverse_temperature"] * reward * chosen,
        jnp.array([latents[4], p["slow_habit_weight"], 0.0])])
    return latents + rates * (targets - latents)
# EVOLVE-BLOCK-END