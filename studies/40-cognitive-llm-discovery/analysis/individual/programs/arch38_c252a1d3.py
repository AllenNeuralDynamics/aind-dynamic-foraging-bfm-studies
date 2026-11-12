# EVOLVE-BLOCK-START
"""Dual-rate Rescorla-Wagner values whose unchosen side relaxes toward an expected accumulated
bait in baited tasks (toward 0 otherwise); counterfactual inference (anti-correlated sides,
separate strength in coupled and uncoupled tasks); fast and slow signed choice (habit) traces;
and a side bias."""
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
    "bait_buildup_rate": (0.0, 0.6),          # subjective per-trial baiting of the unvisited side
    "bait_expectation_weight": (-3.0, 5.0),   # pull toward the side expected to hold a bait
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_habit_signed", "slow_habit_signed",
                "task_coupled", "signed_run_length")

LAPSE = 1e-4


def init_latents(p, session):
    coupled = jnp.asarray(session["coupled"], dtype=jnp.float64)
    z = 0.0 * coupled
    return jnp.stack([z, z, z, z, coupled, z])


def predict(latents, p):
    q, h = latents[:2], latents[2:4]
    run = latents[5]
    run_sign = jnp.sign(run)
    n_away = jnp.abs(run)
    # Expected accumulated bait on the other side after n trials away from it.
    bait = 1.0 - jnp.power(1.0 - p["bait_buildup_rate"], n_away)
    drive = (p["value_inverse_temperature"] * (q[1] - q[0])
             + p["fast_habit_weight"] * h[0]
             + p["slow_habit_weight"] * h[1]
             - p["bait_expectation_weight"] * run_sign * bait
             + p["right_bias"])
    p_right = LAPSE + (1.0 - 2.0 * LAPSE) * jax.nn.sigmoid(drive)
    return jnp.log(jnp.stack([1.0 - p_right, p_right]))


def update(latents, choice, reward, p):
    q = latents[:2]
    coupled, run = latents[4], latents[5]
    chosen = jax.nn.one_hot(choice, 2)
    rate = jnp.where(reward > 0, p["learn_rate_rewarded"], p["learn_rate_unrewarded"])
    learned = q + rate * (reward - q)
    # Unchosen side: value is forgotten (decays toward 0).
    forgotten = (1.0 - p["forget_rate_unchosen"]) * q
    # Coupled tasks: a miss on the chosen side implies the other side pays.
    cf_weight = coupled * p["counterfactual_weight"]
    counterfactual = cf_weight * rate * ((1.0 - reward) - forgotten)
    q = jnp.where(chosen > 0, learned, forgotten + counterfactual)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_habit_step_size"] * (signed - latents[2])
    slow = latents[3] + p["slow_habit_step_size"] * (signed - latents[3])
    # Run length on the current side (= trials the other side has been left alone).
    same_side = jnp.sign(run) == signed
    n = jnp.where(same_side, jnp.abs(run) + 1.0, 1.0)
    run = signed * n
    return jnp.stack([q[0], q[1], fast, slow, coupled, run])
# EVOLVE-BLOCK-END