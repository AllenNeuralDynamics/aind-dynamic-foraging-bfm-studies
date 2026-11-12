# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values with forgetting of the unchosen side toward 0,
fast and slow signed choice (habit) traces whose fast component's weight changes with
training experience (log number of sessions), and a side bias."""
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
LATENT_NAMES = ("q_left_scaled_negative", "q_right_scaled", "fast_habit_weighted",
                "slow_habit_weighted", "fast_habit_weight_this_session")


def init_latents(p, session):
    # Fast-habit weight for this session: grows/shrinks with log(1 + sessions), centred ~20.
    fast_weight = p["fast_habit_weight"] + p["habit_experience_gain"] * (
        jnp.log1p(session["index"]) - 3.0)
    return jnp.array([0.0, 0.0, 0.0, 0.0, fast_weight])


def predict(latents, p):
    # Latents already carry their decision weights and signs (left negative),
    # so the right-choice logit is a plain sum.
    return jnp.array([0.0, jnp.sum(latents[:4]) + p["right_bias"]])


def update(latents, choice, reward, p):
    # One delta rule for every latent: state moves toward its target at its own rate.
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    # Chosen value learns toward (scaled) reward; unchosen value is forgotten toward 0;
    # habit traces move toward the weighted signed choice; session weight stays fixed.
    rates = jnp.concatenate([
        jnp.where(chosen, rate, p["forget_rate_unchosen"]),
        jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"], 0.0])])
    # All targets share the signed choice (+0.5 right, -0.5 left).
    targets = (choice - 0.5) * jnp.concatenate([
        p["value_inverse_temperature"] * chosen * reward,
        jnp.array([latents[4], p["slow_habit_weight"], 0.0])])
    return latents + rates * (targets - latents)
# EVOLVE-BLOCK-END