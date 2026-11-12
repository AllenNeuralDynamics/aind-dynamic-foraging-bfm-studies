"""Template: a side bias and a tendency to repeat the last choice; no learning from reward."""
# EVOLVE-BLOCK-START
import jax.numpy as jnp

PARAMS = {
    "learn_rate_rew": (0.0, 1.0),        # value update after a reward
    "learn_rate_unrew": (0.0, 1.0),      # value update after no reward
    "forget_rate": (0.0, 1.0),           # decay of the unchosen side's value towards 0
    "beta": (0.0, 20.0),                 # inverse temperature on the value difference
    "fast_trace_step": (0.0, 1.0),       # step size of the fast choice trace (perseveration)
    "fast_trace_weight": (-3.0, 5.0),    # pull of the fast trace, in logits
    "slow_trace_step": (0.0, 0.3),       # step size of the slow choice trace (long-run habit)
    "slow_trace_weight": (-3.0, 3.0),    # pull of the slow trace, in logits
    "side_bias": (-3.0, 3.0),            # fixed preference for the right side, in logits
    "counterfactual_weight": (0.0, 1.0), # coupled tasks: unchosen side learns from 1 - reward
    "reward_rate_step": (0.0, 0.5),      # step size of the running average reward rate
    "rate_perseveration_gain": (-6.0, 6.0),  # change of fast-trace weight per unit reward rate
}
N_LATENT = 6
LATENT_NAMES = ("q_left", "q_right", "fast_trace_signed", "slow_trace_signed", "task_coupled",
                "reward_rate")
RATE_CENTER = 0.5


def init_latents(p, session):
    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0 * session["coupled"], RATE_CENTER])


def predict(latents, p):
    beta = p["beta"]
    stay_weight = p["fast_trace_weight"] + p["rate_perseveration_gain"] * (latents[5] - RATE_CENTER)
    right_logit = (beta * (latents[1] - latents[0])
                   + stay_weight * latents[2]
                   + p["slow_trace_weight"] * latents[3]
                   + p["side_bias"])
    return jnp.stack([-0.5 * right_logit, 0.5 * right_logit])


def update(latents, choice, reward, p):
    q = latents[:2]
    chosen = jnp.array([1.0 - choice, choice * 1.0])
    rate = jnp.where(reward > 0, p["learn_rate_rew"], p["learn_rate_unrew"])
    learned = q + rate * (reward - q)
    forgotten = (1.0 - p["forget_rate"]) * q
    # In coupled tasks the unchosen side is inferred to be the opposite of the observed outcome.
    coupled = latents[4]
    counterfactual = coupled * p["counterfactual_weight"] * rate * ((1.0 - reward) - forgotten)
    unchosen = forgotten + counterfactual
    q = jnp.where(chosen > 0, learned, unchosen)
    signed = 2.0 * choice - 1.0
    fast = latents[2] + p["fast_trace_step"] * (signed - latents[2])
    slow = latents[3] + p["slow_trace_step"] * (signed - latents[3])
    rbar = latents[5] + p["reward_rate_step"] * (reward - latents[5])
    return jnp.array([q[0], q[1], fast, slow, coupled, rbar])
# EVOLVE-BLOCK-END