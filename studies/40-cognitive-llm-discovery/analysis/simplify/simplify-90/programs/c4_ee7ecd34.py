# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values; the unchosen side decays (retention) and, after a miss,
is pushed toward 'the other side pays' (strength depends on coupled vs uncoupled task).
Fast and slow habit drives that chase +weight after a right choice and -weight after a left
choice; the fast weight changes with log training experience. Plus a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),
    "learn_rate_unrewarded": (0.0, 1.0),
    "retain_unchosen": (0.0, 1.0),        # fraction of unchosen value kept each trial
    "value_inverse_temperature": (0.0, 20.0),
    "fast_habit_step_size": (0.2, 1.0),
    "fast_habit_weight": (-4.0, 5.0),     # fast-habit weight on the first session
    "slow_habit_step_size": (0.0, 0.3),
    "slow_habit_weight": (-3.0, 3.0),
    "right_bias": (-3.0, 3.0),
    "counterfactual_weight": (0.0, 1.0),
    "counterfactual_weight_uncoupled": (0.0, 1.0),
    "habit_experience_gain": (-1.5, 1.5),  # change of fast-habit weight per log(sessions)
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_drive", "slow_habit_drive",
                "counterfactual_strength", "fast_habit_weight_now")


def init_latents(p, session):
    # Session constants: counterfactual strength for this task type, and the fast-habit
    # weight at this point in training.
    cf = jnp.where(session["coupled"], p["counterfactual_weight"],
                   p["counterfactual_weight_uncoupled"])
    w = p["fast_habit_weight"] + p["habit_experience_gain"] * jnp.log1p(session["index"])
    return jnp.array([0.0, 0.0, 0.0, 0.0, cf, w])


def predict(latents, p):
    # Right-choice logit: value difference + habit drives (weights included) + side bias.
    return jnp.array([0.0, p["value_inverse_temperature"] * (latents[1] - latents[0])
                      + latents[2] + latents[3] + p["right_bias"]])


def update(latents, choice, reward, p):
    q, h, const = latents[:2], latents[2:4], latents[4:]
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    # Unchosen side decays; a miss on the chosen side implies the other side pays.
    kept = p["retain_unchosen"] * q
    q = jnp.where(jax.nn.one_hot(choice, 2), q + rate * (reward - q),
                  kept + const[0] * rate * (1.0 - reward - kept))
    # Habit drives chase +weight (right choice) or -weight (left choice).
    w = jnp.array([const[1], p["slow_habit_weight"]])
    h = h + jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"]]) * (
        jnp.where(choice, w, -w) - h)
    return jnp.concatenate([q, h, const])
# EVOLVE-BLOCK-END