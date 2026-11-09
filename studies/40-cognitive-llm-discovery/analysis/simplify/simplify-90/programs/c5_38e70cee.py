# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values; the unchosen side decays (retention) and gets a
counterfactual update (anti-correlated sides; strength depends on coupled vs uncoupled task);
fast and slow signed choice (habit) traces (+0.5 right, -0.5 left), the fast trace's weight
changing with training experience; and a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),
    "learn_rate_unrewarded": (0.0, 1.0),
    "retain_rate_unchosen": (0.0, 1.0),     # fraction of unchosen value kept each trial
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
LATENT_NAMES = ("q_left", "q_right", "fast_habit_signed", "slow_habit_signed",
                "counterfactual_strength", "fast_habit_weight_now")


def init_latents(p, session):
    # Counterfactual strength depends on the task type (fixed within a session).
    cf = jnp.where(session["coupled"], p["counterfactual_weight"],
                   p["counterfactual_weight_uncoupled"])
    # Fast-habit weight at this point of training (changes with log(1 + sessions)).
    w = p["fast_habit_weight"] + p["habit_experience_gain"] * jnp.log1p(session["index"])
    return jnp.array([0.0, 0.0, 0.0, 0.0, cf, w])


def predict(latents, p):
    # Habit traces already carry their weights.
    return jnp.array([0.0, p["value_inverse_temperature"] * (latents[1] - latents[0])
                      + latents[2] + latents[3] + p["right_bias"]])


def update(latents, choice, reward, p):
    q = latents[:2]
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    # Unchosen side decays toward 0; a miss on the chosen side implies the other side pays.
    kept = p["retain_rate_unchosen"] * q
    q = jnp.where(jax.nn.one_hot(choice, 2), q + rate * (reward - q),
                  kept + latents[4] * rate * (1.0 - reward - kept))
    # Fast and slow choice traces moving toward weight * (+0.5 right / -0.5 left).
    h = latents[2:4] + jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"]]) * (
        jnp.array([latents[5], p["slow_habit_weight"]]) * (choice - 0.5) - latents[2:4])
    return jnp.concatenate([q, h, latents[4:]])
# EVOLVE-BLOCK-END