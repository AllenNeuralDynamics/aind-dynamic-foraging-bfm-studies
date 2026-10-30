"""Shared pieces of I4–I6: transition groups, per-group affine fits, the 1-variable model.

A transition is row t → t+1 within a session. Its group is the input received in between —
trial t's choice × reward — and whether trial t repeated trial t−1's side ("stay"), switched
("switch"), or had no previous trial ("first"): 12 groups, indexed
    g = 3·(2·choice + reward) + {0 stay, 1 switch, 2 first}.
A row's readout cluster is its *last* input (trial t−1's choice × reward) or 4 for a first trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_NAMES = ["L unrewarded", "L rewarded", "R unrewarded", "R rewarded"]  # index 2·choice + reward
KINDS = ["stay", "switch", "first"]
GROUP_NAMES = [f"{i} ({k})" for i in INPUT_NAMES for k in KINDS]
N_GROUPS, N_CLUSTERS = len(GROUP_NAMES), 5
MIN_PER_GROUP = 50


def session_layout(t: pd.DataFrame) -> dict:
    """Arrays describing transitions and readout clusters for a trial table (rows in session order)."""
    ses = t["ses_idx"].to_numpy()
    action = t["action"].to_numpy().astype(int)
    reward = (t["reward"].to_numpy() > 0).astype(int)
    first = np.r_[True, ses[1:] != ses[:-1]]
    last = np.r_[ses[1:] != ses[:-1], True]
    prev_action = np.r_[-1, action[:-1]]
    prev_action[first] = -1
    kind = np.where(first, 2, np.where(action == prev_action, 0, 1))
    group = 3 * (2 * action + reward) + kind  # group of the transition leaving row t
    cluster = np.where(first, 4, 2 * np.r_[0, action[:-1]] + np.r_[0, reward[:-1]])
    return {"first": first, "has_next": ~last, "group": group, "cluster": cluster}


def fit_groups(v: np.ndarray, layout: dict, rows: np.ndarray | None = None, fallback: np.ndarray | None = None):
    """Per-group least-squares v(t+1) = c + λ v(t); returns (λ [G], v* [G], n [G]).

    v* = c / (1 − λ). Groups with fewer than MIN_PER_GROUP transitions take ``fallback``
    (λ, v*) rows when given, else NaN.
    """
    idx = np.flatnonzero(layout["has_next"]) if rows is None else rows[layout["has_next"][rows]]
    lam, target, n = np.full(N_GROUPS, np.nan), np.full(N_GROUPS, np.nan), np.zeros(N_GROUPS, int)
    for g in range(N_GROUPS):
        sel = idx[layout["group"][idx] == g]
        n[g] = len(sel)
        if n[g] >= MIN_PER_GROUP and g % 3 == 2:
            # First transitions all start from (nearly) the same state, so a slope is not
            # identified: fit the constant v(t+1) instead (λ = 0, v* = its mean).
            lam[g], target[g] = 0.0, float(v[sel + 1].mean())
        elif n[g] >= MIN_PER_GROUP and np.ptp(v[sel]) > 0:
            slope, intercept = np.polyfit(v[sel], v[sel + 1], 1)
            lam[g], target[g] = slope, intercept / (1 - slope) if slope != 1 else np.nan
        elif fallback is not None:
            lam[g], target[g] = fallback[0][g], fallback[1][g]
    return lam, target, n


def fit_readout(v: np.ndarray, logit: np.ndarray, layout: dict, rows: np.ndarray | None = None):
    """Least squares logit ≈ a·v + b[cluster]; returns (a, b [N_CLUSTERS])."""
    rows = np.arange(len(v)) if rows is None else rows
    design = np.column_stack([v[rows], np.eye(N_CLUSTERS)[layout["cluster"][rows]]])
    coef, *_ = np.linalg.lstsq(design, logit[rows], rcond=None)
    return float(coef[0]), coef[1:]


def simulate(layout: dict, v0: float, lam_rows: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    """Run v(t+1) = v* + λ (v(t) − v*) along every session from v0, driven by the observed inputs.

    lam_rows / target_rows give the parameters for each row's outgoing transition (so per-mouse
    parameters can be supplied row-wise). Vectorised over sessions, looping over trial index.
    """
    n = len(layout["group"])
    v = np.empty(n)
    starts = np.flatnonzero(layout["first"])
    lengths = np.diff(np.r_[starts, n])
    v[starts] = v0
    for k in range(1, lengths.max()):
        rows = starts[lengths > k] + k
        prev = rows - 1
        v[rows] = target_rows[prev] + lam_rows[prev] * (v[prev] - target_rows[prev])
    return v


def likelihood(action: np.ndarray, logit: np.ndarray) -> float:
    p = 1 / (1 + np.exp(-logit))
    p = np.where(action == 1, p, 1 - p)
    return float(np.exp(np.mean(np.log(np.clip(p, 1e-10, 1 - 1e-10)))))


def fit_groups_nd(x: np.ndarray, layout: dict, rows: np.ndarray | None = None, fallback=None):
    """Per-group least squares x(t+1) = A x(t) + c for a d-dim carried state x [N, d].

    Returns (A [G, d, d], c [G, d], n [G]). First-trial groups are fitted as constants (A = 0),
    as in ``fit_groups``. Groups with fewer than MIN_PER_GROUP transitions take ``fallback``
    (A, c) when given, else NaN.
    """
    d = x.shape[1]
    idx = np.flatnonzero(layout["has_next"]) if rows is None else rows[layout["has_next"][rows]]
    A, c, n = np.full((N_GROUPS, d, d), np.nan), np.full((N_GROUPS, d), np.nan), np.zeros(N_GROUPS, int)
    for g in range(N_GROUPS):
        sel = idx[layout["group"][idx] == g]
        n[g] = len(sel)
        if n[g] >= MIN_PER_GROUP and g % 3 == 2:
            A[g], c[g] = 0.0, x[sel + 1].mean(axis=0)
        elif n[g] >= MIN_PER_GROUP:
            design = np.column_stack([x[sel], np.ones(len(sel))])
            coef, *_ = np.linalg.lstsq(design, x[sel + 1], rcond=None)
            A[g], c[g] = coef[:d].T, coef[d]
        elif fallback is not None:
            A[g], c[g] = fallback[0][g], fallback[1][g]
    return A, c, n


def fit_readout_nd(x: np.ndarray, logit: np.ndarray, layout: dict, rows: np.ndarray | None = None):
    """Least squares logit ≈ x·a + b[cluster] for x [N, d]; returns (a [d], b [N_CLUSTERS])."""
    rows = np.arange(len(x)) if rows is None else rows
    design = np.column_stack([x[rows], np.eye(N_CLUSTERS)[layout["cluster"][rows]]])
    coef, *_ = np.linalg.lstsq(design, logit[rows], rcond=None)
    return coef[:x.shape[1]], coef[x.shape[1]:]


def simulate_nd(layout: dict, x0: np.ndarray, A_rows: np.ndarray, c_rows: np.ndarray) -> np.ndarray:
    """``simulate`` for a d-dim state: x(t+1) = A x(t) + c with row-wise A [N, d, d], c [N, d]."""
    n = len(layout["group"])
    x = np.empty((n, len(x0)))
    starts = np.flatnonzero(layout["first"])
    lengths = np.diff(np.r_[starts, n])
    x[starts] = x0
    for k in range(1, lengths.max()):
        rows = starts[lengths > k] + k
        prev = rows - 1
        x[rows] = np.einsum("nij,nj->ni", A_rows[prev], x[prev]) + c_rows[prev]
    return x
