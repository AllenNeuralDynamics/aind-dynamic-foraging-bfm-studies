# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values with forgetting of the unchosen side toward 0,
fast and slow signed choice (habit) traces whose fast component's weight changes with
training experience (log number of sessions), and a side bias.
All latents are kept in logit units, so the right logit is a plain sum."""
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
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit", "slow_habit", "right_bias",
                "fast_habit_weight_this_session")


def init_latents(p, session):
    # Fast-habit weight for this session: changes with log(1 + sessions), centred ~20 sessions.
    fast_weight = p["fast_habit_weight"] + p["habit_experience_gain"] * (
        jnp.log1p(session["index"]) - 3.0)
    return jnp.array([0.0, 0.0, 0.0, 0.0, p["right_bias"], fast_weight])


def predict(latents, p):
    # Right logit = right value + fast habit + slow habit + bias.
    return jnp.array([latents[0], jnp.sum(latents[1:5])])


def update(latents, choice, reward, p):
    # One delta rule: every latent moves toward its target at its own rate.
    chosen = jax.nn.one_hot(choice, 2)
    # Chosen value -> beta*reward (rate depends on reward); unchosen value forgotten toward 0;
    # habit traces -> weight * (+0.5 right, -0.5 left); bias and session weight fixed (rate 0).
    rates = jnp.concatenate([
        jnp.where(chosen,
                  jnp.where(reward, p["learn_rate_rewarded"], p["learn_rate_unrewarded"]),
                  p["forget_rate_unchosen"]),
        jnp.array([p["fast_habit_step_size"], p["slow_habit_step_size"], 0.0, 0.0])])
    targets = jnp.concatenate([
        chosen * reward * p["value_inverse_temperature"],
        (choice - 0.5) * jnp.array([latents[5], p["slow_habit_weight"], 0.0, 0.0])])
    return latents + rates * (targets - latents)
# EVOLVE-BLOCK-END