# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values whose unchosen side relaxes toward a task-dependent target
(an expected accumulated bait in baited tasks, 0 otherwise) plus counterfactual inference in
coupled tasks; fast and slow signed choice (habit) traces and a side bias."""
import jax
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rewarded": (0.0, 1.0),
    "learn_rate_unrewarded": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "value_inverse_temperature": (0.0, 20.0),
    "fast_habit_step_size": (0.2, 1.0),
    "fast_habit_weight": (-3.0, 5.0),
    "slow_habit_step_size": (0.0, 0.3),
    "slow_habit_weight": (-3.0, 3.0),
    "right_bias": (-3.0, 3.0),
    "counterfactual_weight": (0.0, 1.0),
    "reward_seeking_step": (0.0, 1.0),     # step size of win-stay/lose-shift trace
    "reward_seeking_weight": (-5.0, 5.0),  # weight of win-stay/lose-shift trace
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_signed", "slow_habit_signed",
                "task_coupled", "reward_seeking_trace")


def init_latents(p, session):
    coupled = jnp.asarray(session["coupled"], dtype=jnp.float64)
    z = 0.0 * coupled
    return jnp.stack([z, z, z, z, coupled, z])


def predict(latents, p):
    q, h = latents[:2], latents[2:4]
    right_logit = (p["value_inverse_temperature"] * (q[1] - q[0])
                   + p["fast_habit_weight"] * h[0]
                   + p["slow_habit_weight"] * h[1]
                   + p["reward_seeking_weight"] * latents[5]
                   + p["right_bias"])
    return jnp.stack([jnp.zeros_like(right_logit), right_logit])


def update(latents, choice, reward, p):
    q = latents[:2]
    coupled, seeking = latents[4], latents[5]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    learned = q + rate * (reward - q)
    # Unchosen side: forgotten toward 0.
    forgotten = (1.0 - p["forget_rate_unchosen"]) * q
    # Coupled tasks: a miss on the chosen side implies the other side pays.
    counterfactual = coupled * p["counterfactual_weight"] * rate * ((1.0 - reward) - forgotten)
    q = jnp.where(chosen > 0, learned, forgotten + counterfactual)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_habit_step_size"] * (signed - latents[2])
    slow = latents[3] + p["slow_habit_step_size"] * (signed - latents[3])
    signed_feedback = 2.0 * reward - 1.0
    seeking = seeking + p["reward_seeking_step"] * (signed * signed_feedback - seeking)
    return jnp.stack([q[0], q[1], fast, slow, coupled, seeking])
# EVOLVE-BLOCK-END