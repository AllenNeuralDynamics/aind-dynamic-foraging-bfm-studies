# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values with partial retention of the unchosen side and counterfactual
inference (anti-correlated sides; separate strength in coupled and uncoupled tasks), fast and
slow signed choice (habit) traces whose fast component's weight grows or shrinks with
log training experience, and a side bias."""
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
    # weight at this point in training (grows with log number of sessions).
    cf = jnp.where(session["coupled"], p["counterfactual_weight"],
                   p["counterfactual_weight_uncoupled"])
    w = p["fast_habit_weight"] + p["habit_experience_gain"] * jnp.log1p(session["index"])
    return jnp.array([0.0, 0.0, 0.0, 0.0, cf, w])


def predict(latents, p):
    # Habit traces already carry their weights.
    right_logit = (p["value_inverse_temperature"] * (latents[1] - latents[0])
                   + latents[2] + latents[3] + p["right_bias"])
    return jnp.array([0.0, right_logit])


def update(latents, choice, reward, p):
    q, h = latents[:2], latents[2:4]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    # Unchosen side decays toward 0; a miss on the chosen side implies the other pays.
    kept = p["unchosen_retention"] * q
    # Each side: base value + learning rate * its prediction error.
    q = jnp.where(chosen, q, kept) + rate * jnp.where(
        chosen, reward - q, latents[4] * (1.0 - reward - kept))
    # Fast and slow choice traces moving toward weight * (+0.5 right / -0.5 left).
    target = jnp.array([latents[5], p["slow_habit_weight"]]) * (choice - 0.5)
    h = h + jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"]]) * (target - h)
    return jnp.concatenate([q, h, latents[4:]])
# EVOLVE-BLOCK-END