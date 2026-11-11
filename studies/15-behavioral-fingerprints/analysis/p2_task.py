"""The gym task a P2 rollout plays, from a ``p2_task_specs.json`` entry (ADR 0006).

Kept apart from ``freeze_p2_tasks.py``, which needs the database, so the rollout workers import
only the gym.
"""
from __future__ import annotations

PERSEVERATIVE_LIMIT, MAX_BLOCK_TALLY = 4, 3   # the real uncoupled task's values (the gym's tally is 4)


def _held_coupled_task():
    """``CoupledBlockTask`` with the real task's minimum-reward hold (``BlockMinReward``): a block
    that is due to end is extended one trial at a time until it has paid ``min_reward`` rewards.
    Only for the holds check (README, P2); ADR 0006's rollouts have no hold."""
    from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask

    class HeldCoupledBlockTask(CoupledBlockTask):
        def __init__(self, min_reward: int, **kwargs):
            super().__init__(**kwargs)
            self.min_reward = int(min_reward)

        def generate_new_trial(self):
            if self.block_lens and self.trial == self.block_starts[-1]:
                start = self.block_starts[-2]
                if self.reward[start:self.trial].sum() < self.min_reward:   # hold: one more trial
                    self.block_starts[-1] += 1
                    self.block_lens[-1] += 1
            return super().generate_new_trial()

    return HeldCoupledBlockTask


def build_task(spec: dict, seed: int, hold: bool = False):
    """The gym task of a spec: coupled (and warm-up) or uncoupled, with the session's parameters;
    ``hold`` adds the minimum-reward hold to a coupled task that records one."""
    from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask, UncoupledBlockTask

    common = dict(reward_baiting=spec["baited"], num_trials=spec["n_trials"], seed=seed)
    if spec["kind"] == "uncoupled":
        return UncoupledBlockTask(rwd_prob_array=spec["reward_probabilities"], block_min=int(spec["block_min"]),
                                  block_max=int(spec["block_max"]), perseverative_limit=PERSEVERATIVE_LIMIT,
                                  max_block_tally=MAX_BLOCK_TALLY, **common)
    kwargs = dict(block_min=int(spec["block_min"]), block_max=int(spec["block_max"]),
                  block_beta=spec["block_beta"], p_reward_pairs=spec["reward_pairs"], **common)
    if hold:
        return _held_coupled_task()(min_reward=spec["min_reward_each_block"], **kwargs)
    return CoupledBlockTask(**kwargs)
