# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values with partial retention of the unchosen side and
counterfactual inference (separate strength in coupled and uncoupled tasks), plus fast and
slow habit traces that track weighted signed choices (the fast weight changes with training
experience), and a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),
    "learn_rate_unrewarded": (0.0, 1.0),
    "unchosen_retention": (0.0, 1.0),  # fraction of the unchosen value kept each trial
    "value_inverse_temperature": (0.0, 20.0),
    "fast_habit_step_size": (0.2, 1.0),
    "fast_habit_weight": (-8.0, 10.0),  # fast-habit weight on the first session
    "slow_habit_step_size": (0.0, 0.3),
    "slow_habit_weight": (-6.0, 6.0),
    "right_bias": (-3.0, 3.0),
    "counterfactual_weight": (0.0, 1.0),
    "counterfactual_weight_uncoupled": (0.0, 1.0),
    "habit_experience_gain": (-3.0, 3.0),  # change of fast-habit weight per log(sessions)
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_drive", "slow_habit_drive",
                "counterfactual_strength", "fast_habit_weight_now")


def init_latents(p, session):
    # Session constants: counterfactual strength for this task type, and the fast-habit
    # weight at this point in training (centred near ~20 sessions).
    cf = jnp.where(session["coupled"], p["counterfactual_weight"],
                   p["counterfactual_weight_uncoupled"])
    w = p["fast_habit_weight"] + p["habit_experience_gain"] * jnp.log1p(session["index"])
    return jnp.array([0.0, 0.0, 0.0, 0.0, cf, w])


def predict(latents, p):
    # Habit traces already carry their weights, so they add directly to the logit.
    return jnp.array([0.0, p["value_inverse_temperature"] * (latents[1] - latents[0])
                      + latents[2] + latents[3] + p["right_bias"]])


def update(latents, choice, reward, p):
    q = latents[:2]
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    # Unchosen side decays toward 0; a miss on the chosen side implies the other pays.
    kept = p["unchosen_retention"] * q
    q = jnp.where(jax.nn.one_hot(choice, 2), q + rate * (reward - q),
                  kept + latents[4] * rate * (1.0 - reward - kept))
    # Fast and slow traces of weighted signed choice (+w/2 right, -w/2 left).
    h = latents[2:4] + jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"]]) * (
        jnp.array([latents[5], p["slow_habit_weight"]]) * (choice - 0.5) - latents[2:4])
    return jnp.concatenate([q, h, latents[4:]])
# EVOLVE-BLOCK-END