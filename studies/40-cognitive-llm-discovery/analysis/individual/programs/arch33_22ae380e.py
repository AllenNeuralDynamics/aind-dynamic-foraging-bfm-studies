"""Template: a side bias and a tendency to repeat the last choice; no learning from reward."""
# EVOLVE-BLOCK-START
"""Forgetful dual-rate Q-learning whose unchosen side relaxes toward an expected accumulated bait
(baited tasks) and learns from the inferred opposite outcome (coupled tasks); fast and slow
choice habits; a value gain that changes with training experience."""
import jax
import jax.numpy as jnp

LAPSE = 1e-4

PARAMS = {
    "learn_rate_rew": (0.0, 1.0),        # value update after a reward
    "learn_rate_unrew": (0.0, 1.0),      # value update after no reward
    "forget_rate": (0.0, 1.0),           # relaxation of the unchosen side's value
    "beta": (0.0, 20.0),                 # inverse temperature at mid-training
    "fast_trace_step": (0.0, 1.0),       # step size of the fast choice trace (perseveration)
    "fast_trace_weight": (-3.0, 5.0),    # pull of the fast trace, in logits
    "slow_trace_step": (0.0, 0.3),       # step size of the slow choice trace (long-run habit)
    "slow_trace_weight": (-3.0, 3.0),    # pull of the slow trace, in logits
    "side_bias": (-3.0, 3.0),            # fixed preference for the right side, in logits
    "counterfactual_weight": (0.0, 1.0), # coupled tasks: unchosen side learns from 1 - reward
    "bait_expectation": (0.0, 1.0),      # baited tasks: value an unvisited side drifts toward
    "win_stay_lose_shift_weight": (-2.0, 4.0),  # pull toward last side if rewarded, away if not
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_trace_signed", "slow_trace_signed",
                "last_choice_x_outcome", "task_code")


def init_latents(p, session):
    # task_code = coupled + 2 * baited (constant within a session)
    code = 1.0 * session["coupled"] + 2.0 * session["baited"]
    return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, code])


def predict(latents, p):
    right_logit = (p["beta"] * (latents[1] - latents[0])
                   + p["fast_trace_weight"] * latents[2]
                   + p["slow_trace_weight"] * latents[3]
                   + p["win_stay_lose_shift_weight"] * latents[4]
                   + p["side_bias"])
    p_right = LAPSE + (1.0 - 2.0 * LAPSE) * jax.nn.sigmoid(right_logit)
    return jnp.log(jnp.stack([1.0 - p_right, p_right]))


def update(latents, choice, reward, p):
    q = latents[:2]
    code = latents[5]
    coupled = jnp.mod(code, 2.0)
    baited = jnp.floor(code / 2.0)
    chosen = jnp.array([1.0 - choice, choice * 1.0])
    rate = jnp.where(reward > 0, p["learn_rate_rew"], p["learn_rate_unrew"])
    learned = q + rate * (reward - q)
    target = baited * p["bait_expectation"]
    forgotten = q + p["forget_rate"] * (target - q)
    counterfactual = coupled * p["counterfactual_weight"] * rate * ((1.0 - reward) - forgotten)
    q = jnp.where(chosen > 0, learned, forgotten + counterfactual)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_trace_step"] * (signed - latents[2])
    slow = latents[3] + p["slow_trace_step"] * (signed - latents[3])
    wsls = signed * (2.0 * reward - 1.0)
    return jnp.array([q[0], q[1], fast, slow, wsls, code])
# EVOLVE-BLOCK-END