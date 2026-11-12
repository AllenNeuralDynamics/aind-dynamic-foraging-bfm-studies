# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values with forgetting of the unchosen side and counterfactual
inference (anti-correlated sides; separate strength in coupled and uncoupled tasks), fast and
slow signed choice (habit) traces whose fast component's weight changes with training
experience, and a side bias."""
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
    "counterfactual_weight_uncoupled": (0.0, 1.0),
    "satiety_rate": (-1.0, 1.0),  # change of log decision gain per 100 rewards collected
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_signed", "slow_habit_signed",
                "task_coupled", "rewards_collected_per100")

SATIETY_CENTER = 1.0  # 100 rewards: weights describe a mid-session mouse

LAPSE = 1e-4


def init_latents(p, session):
    coupled = jnp.asarray(session["coupled"], dtype=jnp.float64)
    z = 0.0 * coupled
    return jnp.stack([z, z, z, z, coupled, z])


def predict(latents, p):
    q, h = latents[:2], latents[2:4]
    gain = jnp.exp(-p["satiety_rate"] * (latents[5] - SATIETY_CENTER))
    deliberate = (p["value_inverse_temperature"] * (q[1] - q[0])
                  + p["fast_habit_weight"] * h[0]
                  + p["slow_habit_weight"] * h[1])
    right_logit = gain * deliberate + p["right_bias"]
    p_right = LAPSE + (1.0 - 2.0 * LAPSE) * jax.nn.sigmoid(right_logit)
    return jnp.log(jnp.stack([1.0 - p_right, p_right]))


def update(latents, choice, reward, p):
    q = latents[:2]
    coupled = latents[4]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    learned = q + rate * (reward - q)
    # Unchosen side is forgotten toward 0.
    forgotten = (1.0 - p["forget_rate_unchosen"]) * q
    # A miss on the chosen side implies the other side pays (belief in anti-correlated sides).
    cf_weight = (coupled * p["counterfactual_weight"]
                 + (1.0 - coupled) * p["counterfactual_weight_uncoupled"])
    counterfactual = cf_weight * rate * ((1.0 - reward) - forgotten)
    q = jnp.where(chosen > 0, learned, forgotten + counterfactual)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_habit_step_size"] * (signed - latents[2])
    slow = latents[3] + p["slow_habit_step_size"] * (signed - latents[3])
    rewards = latents[5] + 0.01 * reward
    return jnp.stack([q[0], q[1], fast, slow, coupled, rewards])
# EVOLVE-BLOCK-END