# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values whose unchosen side relaxes toward a task-dependent target
(expected accumulated bait in baited tasks, 0 otherwise) plus counterfactual inference in coupled
tasks; fast and slow signed choice (habit) traces and a side bias. Engagement (the gain on all
history-driven terms) changes within the session with a satiety rate."""
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
    "bait_expectation": (0.0, 1.0),
    "satiety_rate": (-1.5, 3.0),
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_signed", "slow_habit_signed",
                "task_code", "trial_count")
SATIETY_SCALE = 500.0  # trials per unit of satiety


def init_latents(p, session):
    coupled = jnp.asarray(session["coupled"], dtype=jnp.float64)
    baited = jnp.asarray(session["baited"], dtype=jnp.float64)
    z = 0.0 * coupled
    code = coupled + 2.0 * baited  # 0..3: coupled bit + 2 * baited bit
    return jnp.stack([z, z, z, z, code, z])


def predict(latents, p):
    q, h = latents[:2], latents[2:4]
    beta = p["value_inverse_temperature"]
    engagement = jnp.exp(-p["satiety_rate"] * latents[5] / SATIETY_SCALE)
    history = (beta * (q[1] - q[0])
               + p["fast_habit_weight"] * h[0]
               + p["slow_habit_weight"] * h[1])
    right_logit = engagement * history + p["right_bias"]
    return jnp.stack([jnp.zeros_like(right_logit), right_logit])


def update(latents, choice, reward, p):
    q = latents[:2]
    code = latents[4]
    baited = jnp.where(code > 1.5, 1.0, 0.0)
    coupled = code - 2.0 * baited
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    learned = q + rate * (reward - q)
    target = baited * p["bait_expectation"]
    forgotten = q + p["forget_rate_unchosen"] * (target - q)
    counterfactual = coupled * p["counterfactual_weight"] * rate * ((1.0 - reward) - forgotten)
    q = jnp.where(chosen > 0, learned, forgotten + counterfactual)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_habit_step_size"] * (signed - latents[2])
    slow = latents[3] + p["slow_habit_step_size"] * (signed - latents[3])
    return jnp.stack([q[0], q[1], fast, slow, code, latents[5] + 1.0])
# EVOLVE-BLOCK-END