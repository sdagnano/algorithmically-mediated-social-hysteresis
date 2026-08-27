"""Structural stress tests and reference-emulating restoration for Model I.

The original normal form is retained as an intentionally transparent exemplar.
This module asks a harder question: does the closed-loop protocol distinguish
monostable lag, slow memory, near-critical response, bistability, alternative
links, state reset, and mechanism ablation?  All outputs remain synthetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas

import simulate as core


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

BLUE = HexColor("#1769AA")
ORANGE = HexColor("#D86F1D")
GREEN = HexColor("#238463")
PURPLE = HexColor("#7651A8")
RED = HexColor("#B33A3A")
GREY = HexColor("#66707A")
LIGHT = HexColor("#D8DEE6")
PALE = HexColor("#F4F6F8")

INDEX = {"M": 1, "L": 2, "C": 3}
PROBIT_SLOPE_SCALE = math.sqrt(2.0 * math.pi) / 4.0
BISECTION_ITERATIONS = 70
TANGENCY_RESIDUAL_TOL = 1e-9
GRID_RESIDUAL_TOL = 1e-10
ROOT_DEDUP_TOL = 2e-5
STABILITY_TOL = 1e-7
RECOVERY_MULTIPLIER = 4.0


def structural_regimes() -> dict[str, core.Parameters]:
    """Parameterizations used in the structural rate comparison."""
    return {
        "fast-subsystem monostable, no memory": replace(
            core.BASE, w_g=0.28, w_m=0.0, w_l=0.0, w_c=0.0
        ),
        "fast-subsystem monostable + slow memory": replace(core.BASE, w_g=0.28),
        "near-critical fast subsystem + memory": replace(core.BASE, w_g=0.39),
        "bistable, no memory": replace(
            core.BASE, w_g=0.45, w_m=0.0, w_l=0.0, w_c=0.0
        ),
        "bistable + memory": core.BASE,
    }


def response(field: float, gain: float, link: str) -> float:
    """Matched-slope logistic or probit response in [0,1]."""
    if link == "logistic":
        return core.logistic(gain * field)
    if link == "probit":
        z = PROBIT_SLOPE_SCALE * gain * field
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    raise ValueError(f"unknown link: {link}")


def advance_general(
    state: np.ndarray,
    alpha: float,
    p: core.Parameters,
    dt: float,
    *,
    link: str = "logistic",
    ablated: frozenset[str] = frozenset(),
    frozen_values: dict[str, float] | None = None,
    recovery_multiplier: float = 1.0,
    noise_sd: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One step with separate mechanism ablation and state-freezing controls."""
    g, m, ell, cog = state
    m_eff = 0.0 if "M" in ablated else m
    l_eff = 0.0 if "L" in ablated else ell
    c_eff = 0.0 if "C" in ablated else cog
    field = (
        p.w_g * g
        + p.w_alpha * alpha
        + p.w_m * alpha * m_eff
        + p.w_l * l_eff
        + p.w_c * c_eff
        - p.threshold
    )
    target = response(field, p.gain, link)
    dg = (target - g) / p.tau_g
    dm = (alpha * (g - m) - p.delta_m * (1.0 - alpha) * m) / p.tau_m
    dl = (
        (p.build_0 + p.build_alpha * alpha) * g * (1.0 - ell)
        - p.decay_l * (1.0 - g) * ell
    ) / p.tau_l
    dc = (
        p.consolidate_c * g * (1.0 - cog)
        - recovery_multiplier * p.recover_c * (1.0 - g) * cog
    ) / p.tau_c
    updated = state + dt * np.array([dg, dm, dl, dc], dtype=float)
    if noise_sd > 0.0:
        if rng is None:
            raise ValueError("rng is required when noise_sd > 0")
        updated[0] += float(rng.normal(0.0, noise_sd * math.sqrt(dt)))
    updated = np.clip(updated, 0.0, 1.0)
    if frozen_values:
        for channel, value in frozen_values.items():
            updated[INDEX[channel]] = value
    return updated


def low_reference_trajectory(
    *,
    p: core.Parameters,
    total_steps: int,
    dt: float,
    link: str = "logistic",
) -> np.ndarray:
    """Contemporaneous stationary-alpha=0 control from the common initial state."""
    states = np.empty((total_steps + 1, 4), dtype=float)
    states[0] = np.array([0.001, 0.0, 0.0, 0.0], dtype=float)
    for index in range(total_steps):
        states[index + 1] = advance_general(states[index], 0.0, p, dt, link=link)
    return states


def simulate_general(
    *,
    p: core.Parameters = core.BASE,
    dwell: float = 50.0,
    n_levels: int = 31,
    dt: float = 0.2,
    link: str = "logistic",
    ablated: frozenset[str] = frozenset(),
    reset_channels: frozenset[str] = frozenset(),
    reset_target: str = "none",
    freeze_after_reset: bool = False,
    recovery_multiplier: float = 1.0,
    noise_sd: float = 0.0,
    seed: int = 0,
    label: str = "baseline",
) -> pd.DataFrame:
    """Run a full path with zeroing or reference-emulating reversal policies.

    ``reset_target='zero'`` is only a channel-zeroing diagnostic.  Under
    ``reset_target='reference'`` selected channels are assigned the state of a
    contemporaneous alpha=0 control; if freezing is requested, they track that
    reference trajectory throughout the descending diagnostic window.
    """
    up = np.linspace(0.0, 1.0, n_levels)
    down = np.linspace(1.0, 0.0, n_levels)[1:]
    schedule = [("up", float(a)) for a in up] + [("down", float(a)) for a in down]
    state = np.array([0.001, 0.0, 0.0, 0.0], dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    reset_done = False
    n_steps = max(1, int(round(dwell / dt)))
    total_steps = len(schedule) * n_steps
    reference_path = low_reference_trajectory(
        p=p, total_steps=total_steps, dt=dt, link=link
    )
    elapsed_steps = 0
    for step, (branch, alpha) in enumerate(schedule):
        reset_here = False
        if branch == "down" and reset_channels and not reset_done:
            for channel in reset_channels:
                if reset_target == "zero":
                    state[INDEX[channel]] = 0.0
                elif reset_target == "reference":
                    state[INDEX[channel]] = reference_path[elapsed_steps, INDEX[channel]]
                else:
                    raise ValueError("reset_target must be 'zero' or 'reference' when channels are reset")
            reset_done = True
            reset_here = True
        recovery = recovery_multiplier if branch == "down" else 1.0
        for _ in range(n_steps):
            next_elapsed = elapsed_steps + 1
            frozen_values = None
            if freeze_after_reset and reset_done:
                frozen_values = {
                    channel: float(reference_path[next_elapsed, INDEX[channel]])
                    for channel in reset_channels
                }
            state = advance_general(
                state,
                alpha,
                p,
                dt,
                link=link,
                ablated=ablated,
                frozen_values=frozen_values,
                recovery_multiplier=recovery,
                noise_sd=noise_sd,
                rng=rng,
            )
            elapsed_steps = next_elapsed
        rows.append(
            {
                "label": label,
                "step": step,
                "branch": branch,
                "alpha": alpha,
                "dwell": dwell,
                "link": link,
                "ablated": "+".join(sorted(ablated)) or "none",
                "reset_channels": "+".join(sorted(reset_channels)) or "none",
                "reset_target": reset_target,
                "freeze_after_reset": freeze_after_reset,
                "recovery_multiplier": recovery_multiplier,
                "reset_applied_here": reset_here,
                "G": state[0],
                "M": state[1],
                "L": state[2],
                "C": state[3],
            }
        )
    return pd.DataFrame(rows)


def metrics(
    frame: pd.DataFrame,
    *,
    p: core.Parameters = core.BASE,
    dt: float = 0.2,
    link: str = "logistic",
) -> dict[str, float]:
    up, down = core.aligned_branches(frame)
    alpha = up.alpha.to_numpy()
    gap = down.G.to_numpy() - up.G.to_numpy()
    final = down.iloc[0][["G", "M", "L", "C"]].to_numpy(dtype=float)
    alpha_max = float(np.max(alpha))
    total_steps = len(frame) * max(1, int(round(float(frame.dwell.iloc[0]) / dt)))
    # Same elapsed time, common initialization, and alpha=0 throughout.
    reference = low_reference_trajectory(
        p=p, total_steps=total_steps, dt=dt, link=link
    )[-1]
    state_distance = float(np.linalg.norm(final - reference) / 2.0)
    return {
        "signed_area": float(np.trapezoid(gap, alpha) / alpha_max),
        "absolute_area": float(np.trapezoid(np.abs(gap), alpha) / alpha_max),
        "max_absolute_gap": float(np.max(np.abs(gap))),
        "endpoint_state_deficit": state_distance,
        "endpoint_G_gap": float(final[0] - reference[0]),
    }


def structural_regime_sweep() -> pd.DataFrame:
    regimes = structural_regimes()
    rows: list[dict[str, float | str]] = []
    for label, p in regimes.items():
        for dwell in (5.0, 20.0, 80.0, 320.0):
            frame = simulate_general(p=p, dwell=dwell, label=label)
            rows.append({"regime": label, "link": "logistic", "dwell": dwell, **metrics(frame, p=p)})
    for link in ("logistic", "probit"):
        for dwell in (20.0, 80.0, 320.0):
            label = f"bistable + memory ({link})"
            frame = simulate_general(p=core.BASE, dwell=dwell, link=link, label=label)
            rows.append({"regime": label, "link": link, "dwell": dwell, **metrics(frame, link=link)})
    return pd.DataFrame(rows)


def stationary_reservoirs(g: float, alpha: float, p: core.Parameters) -> np.ndarray:
    """Exact M, L, C equilibrium values conditional on G and fixed alpha."""
    m_denominator = alpha + p.delta_m * (1.0 - alpha)
    m = alpha * g / m_denominator if m_denominator > 0.0 else 0.0
    build = p.build_0 + p.build_alpha * alpha
    l_denominator = build * g + p.decay_l * (1.0 - g)
    ell = build * g / l_denominator
    c_denominator = p.consolidate_c * g + p.recover_c * (1.0 - g)
    cog = p.consolidate_c * g / c_denominator
    return np.array([m, ell, cog], dtype=float)


def equilibrium_residual(g: float, alpha: float, p: core.Parameters) -> float:
    """Scalar fixed-point residual after exact elimination of M, L, and C."""
    m, ell, cog = stationary_reservoirs(g, alpha, p)
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    return core.logistic(p.gain * field) - g


def equilibrium_residual_grid(g: np.ndarray, alpha: float, p: core.Parameters) -> np.ndarray:
    """Vectorized reduced residual used by resolution and tangency audits."""
    m_denominator = alpha + p.delta_m * (1.0 - alpha)
    m = alpha * g / m_denominator if m_denominator > 0.0 else np.zeros_like(g)
    build = p.build_0 + p.build_alpha * alpha
    l_denominator = build * g + p.decay_l * (1.0 - g)
    ell = build * g / l_denominator
    c_denominator = p.consolidate_c * g + p.recover_c * (1.0 - g)
    cog = p.consolidate_c * g / c_denominator
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    z = np.clip(p.gain * field, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z)) - g


def equilibrium_derivative(g: float, alpha: float, p: core.Parameters) -> float:
    """Analytic derivative of the exactly reduced fixed-point residual."""
    m, ell, cog = stationary_reservoirs(g, alpha, p)
    m_denominator = alpha + p.delta_m * (1.0 - alpha)
    dm = alpha / m_denominator if m_denominator > 0.0 else 0.0
    build = p.build_0 + p.build_alpha * alpha
    l_denominator = build * g + p.decay_l * (1.0 - g)
    dl = build * p.decay_l / (l_denominator**2)
    c_denominator = p.consolidate_c * g + p.recover_c * (1.0 - g)
    dc = p.consolidate_c * p.recover_c / (c_denominator**2)
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    target = core.logistic(p.gain * field)
    field_derivative = p.w_g + p.w_m * alpha * dm + p.w_l * dl + p.w_c * dc
    return p.gain * target * (1.0 - target) * field_derivative - 1.0


def equilibrium_derivative_grid(g: np.ndarray, alpha: float, p: core.Parameters) -> np.ndarray:
    m_denominator = alpha + p.delta_m * (1.0 - alpha)
    m = alpha * g / m_denominator if m_denominator > 0.0 else np.zeros_like(g)
    dm = alpha / m_denominator if m_denominator > 0.0 else 0.0
    build = p.build_0 + p.build_alpha * alpha
    l_denominator = build * g + p.decay_l * (1.0 - g)
    ell = build * g / l_denominator
    dl = build * p.decay_l / (l_denominator**2)
    c_denominator = p.consolidate_c * g + p.recover_c * (1.0 - g)
    cog = p.consolidate_c * g / c_denominator
    dc = p.consolidate_c * p.recover_c / (c_denominator**2)
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    z = np.clip(p.gain * field, -50.0, 50.0)
    target = 1.0 / (1.0 + np.exp(-z))
    field_derivative = p.w_g + p.w_m * alpha * dm + p.w_l * dl + p.w_c * dc
    return p.gain * target * (1.0 - target) * field_derivative - 1.0


def full_rhs(state: np.ndarray, alpha: float, p: core.Parameters) -> np.ndarray:
    """Unclipped four-dimensional vector field used for local stability."""
    g, m, ell, cog = state
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    return np.array(
        [
            (core.logistic(p.gain * field) - g) / p.tau_g,
            (alpha * (g - m) - p.delta_m * (1.0 - alpha) * m) / p.tau_m,
            ((p.build_0 + p.build_alpha * alpha) * g * (1.0 - ell)
             - p.decay_l * (1.0 - g) * ell) / p.tau_l,
            (p.consolidate_c * g * (1.0 - cog)
             - p.recover_c * (1.0 - g) * cog) / p.tau_c,
        ],
        dtype=float,
    )


def full_jacobian(state: np.ndarray, alpha: float, p: core.Parameters) -> np.ndarray:
    """Exact 4x4 Jacobian of the deterministic vector field."""
    g, m, ell, cog = state
    field = (
        p.w_g * g + p.w_alpha * alpha + p.w_m * alpha * m
        + p.w_l * ell + p.w_c * cog - p.threshold
    )
    target = core.logistic(p.gain * field)
    slope = p.gain * target * (1.0 - target)
    build = p.build_0 + p.build_alpha * alpha
    return np.array(
        [
            [
                (slope * p.w_g - 1.0) / p.tau_g,
                slope * p.w_m * alpha / p.tau_g,
                slope * p.w_l / p.tau_g,
                slope * p.w_c / p.tau_g,
            ],
            [
                alpha / p.tau_m,
                -(alpha + p.delta_m * (1.0 - alpha)) / p.tau_m,
                0.0,
                0.0,
            ],
            [
                (build * (1.0 - ell) + p.decay_l * ell) / p.tau_l,
                0.0,
                -(build * g + p.decay_l * (1.0 - g)) / p.tau_l,
                0.0,
            ],
            [
                (p.consolidate_c * (1.0 - cog) + p.recover_c * cog) / p.tau_c,
                0.0,
                0.0,
                -(p.consolidate_c * g + p.recover_c * (1.0 - g)) / p.tau_c,
            ],
        ],
        dtype=float,
    )


def _bisect(
    function, left: float, right: float, iterations: int = BISECTION_ITERATIONS
) -> float:
    f_left = float(function(left))
    for _ in range(iterations):
        midpoint = 0.5 * (left + right)
        f_midpoint = float(function(midpoint))
        if f_left * f_midpoint <= 0.0:
            right = midpoint
        else:
            left, f_left = midpoint, f_midpoint
    return 0.5 * (left + right)


def equilibrium_roots(
    alpha: float,
    p: core.Parameters,
    root_grid_points: int = 20001,
) -> list[float]:
    """Enumerate roots with sign brackets and explicit tangency safeguards."""
    grid = np.linspace(1e-10, 1.0 - 1e-10, root_grid_points)
    values = equilibrium_residual_grid(grid, alpha, p)
    derivatives = equilibrium_derivative_grid(grid, alpha, p)
    roots: list[float] = []
    for index in np.where(values[:-1] * values[1:] < 0.0)[0]:
        roots.append(_bisect(lambda x: equilibrium_residual(x, alpha, p), float(grid[index]), float(grid[index + 1])))
    # A saddle-node root can be tangent and have no residual sign change. Locate
    # every critical point of the reduced residual and test it explicitly.
    for index in np.where(derivatives[:-1] * derivatives[1:] < 0.0)[0]:
        critical = _bisect(
            lambda x: equilibrium_derivative(x, alpha, p),
            float(grid[index]),
            float(grid[index + 1]),
        )
        if abs(equilibrium_residual(critical, alpha, p)) < TANGENCY_RESIDUAL_TOL:
            roots.append(critical)
    for index in np.where(np.abs(values) < GRID_RESIDUAL_TOL)[0]:
        roots.append(float(grid[index]))
    roots.sort()
    deduplicated: list[float] = []
    for root in roots:
        if not deduplicated or abs(root - deduplicated[-1]) > ROOT_DEDUP_TOL:
            deduplicated.append(root)
    return deduplicated


def classify_full_system(
    *, alpha_grid_points: int = 401, root_grid_points: int = 20001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan fixed-alpha equilibria and test stability in all four states.

    At equilibrium the three reservoir equations are solvable exactly given G.
    We therefore enumerate the reduced scalar roots over a dense alpha grid and
    evaluate eigenvalues of the full 4x4 Jacobian at every resulting state.
    """
    rows: list[dict[str, float | int | str | bool]] = []
    summaries: list[dict[str, float | int | str]] = []
    alpha_grid = np.linspace(0.0, 1.0, alpha_grid_points)
    for label, p in structural_regimes().items():
        regime_rows: list[dict[str, float | int | str | bool]] = []
        for alpha in alpha_grid:
            for root_index, g in enumerate(equilibrium_roots(float(alpha), p, root_grid_points)):
                reservoirs = stationary_reservoirs(g, float(alpha), p)
                state = np.r_[g, reservoirs]
                eigenvalues = np.linalg.eigvals(full_jacobian(state, float(alpha), p))
                dominant = float(np.max(eigenvalues.real))
                row = {
                    "regime": label,
                    "alpha": float(alpha),
                    "root_index": root_index,
                    "G": g,
                    "M": reservoirs[0],
                    "L": reservoirs[1],
                    "C": reservoirs[2],
                    "dominant_real_eigenvalue": dominant,
                    "stable": dominant < -STABILITY_TOL,
                }
                rows.append(row)
                regime_rows.append(row)
        stable_counts = {
            float(alpha): sum(
                int(row["stable"])
                for row in regime_rows
                if float(row["alpha"]) == float(alpha)
            )
            for alpha in alpha_grid
        }
        multistable_alpha = [alpha for alpha, count in stable_counts.items() if count > 1]
        stable_gaps = [
            -float(row["dominant_real_eigenvalue"])
            for row in regime_rows if bool(row["stable"])
        ]
        max_stable = max(stable_counts.values())
        summaries.append(
            {
                "regime": label,
                "classification": (
                    "full-system multistable on the equilibrium/stability grid"
                    if max_stable > 1 else
                    "full-system monostable on the equilibrium/stability grid"
                ),
                "max_stable_equilibria": max_stable,
                "multistable_alpha_min": min(multistable_alpha) if multistable_alpha else math.nan,
                "multistable_alpha_max": max(multistable_alpha) if multistable_alpha else math.nan,
                "minimum_stable_spectral_gap": min(stable_gaps),
                "alpha_grid_points": len(alpha_grid),
                "root_grid_points": root_grid_points,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def refinement_audit() -> pd.DataFrame:
    """Check that regime classifications survive a doubled resolution audit."""
    summaries = []
    for alpha_points, root_points in ((201, 10001), (401, 20001)):
        _, frame = classify_full_system(
            alpha_grid_points=alpha_points, root_grid_points=root_points
        )
        frame = frame.copy()
        frame["audit_alpha_grid_points"] = alpha_points
        frame["audit_root_grid_points"] = root_points
        summaries.append(frame)
    combined = pd.concat(summaries, ignore_index=True)
    labels = combined.pivot(index="regime", columns="audit_alpha_grid_points", values="classification")
    if not (labels[201] == labels[401]).all():
        raise AssertionError("full-system regime classification changed under grid refinement")
    return combined


def intervention_comparison() -> pd.DataFrame:
    specifications = [
        ("none", frozenset(), "none", False, 1.0, frozenset()),
        ("ablate M", frozenset(), "none", False, 1.0, frozenset({"M"})),
        ("ablate L", frozenset(), "none", False, 1.0, frozenset({"L"})),
        ("ablate M+L", frozenset(), "none", False, 1.0, frozenset({"M", "L"})),
        ("zero M once", frozenset({"M"}), "zero", False, 1.0, frozenset()),
        ("zero L once", frozenset({"L"}), "zero", False, 1.0, frozenset()),
        ("zero M+L once", frozenset({"M", "L"}), "zero", False, 1.0, frozenset()),
        ("reference-restore M+L", frozenset({"M", "L"}), "reference", True, 1.0, frozenset()),
        ("facilitate C recovery", frozenset(), "none", False, RECOVERY_MULTIPLIER, frozenset()),
        ("reference-restore M+L + C recovery", frozenset({"M", "L"}), "reference", True, RECOVERY_MULTIPLIER, frozenset()),
    ]
    rows = []
    trajectories = []
    for label, reset, reset_target, freeze, recovery, ablated in specifications:
        frame = simulate_general(
            dwell=50.0,
            label=label,
            reset_channels=reset,
            reset_target=reset_target,
            freeze_after_reset=freeze,
            recovery_multiplier=recovery,
            ablated=ablated,
        )
        trajectories.append(frame)
        rows.append(
            {
                "intervention": label,
                "operation": "ablation" if ablated else (
                    "platform restoration" if reset else (
                        "human recovery facilitation" if recovery > 1 else "none"
                    )
                ),
                **metrics(frame),
            }
        )
    pd.concat(trajectories, ignore_index=True).to_csv(
        DATA / "model_i_intervention_trajectories.csv", index=False
    )
    return pd.DataFrame(rows)


def stochastic_link_stress(n_seeds: int = 64) -> pd.DataFrame:
    rows = []
    for link in ("logistic", "probit"):
        for seed in range(270000, 270000 + n_seeds):
            frame = simulate_general(
                dwell=50.0,
                link=link,
                noise_sd=0.012,
                seed=seed,
                label=f"{link} + process noise",
            )
            rows.append({"link": link, "seed": seed, **metrics(frame, link=link)})
    return pd.DataFrame(rows)


def _rk45_constant_control(
    state: np.ndarray,
    alpha: float,
    duration: float,
    p: core.Parameters,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Independent adaptive Dormand-Prince 5(4) integrator for solver QA."""
    t = 0.0
    h = min(0.2, duration)
    y = state.astype(float).copy()
    while t < duration - 1e-14:
        h = min(h, duration - t)
        f = lambda value: full_rhs(value, alpha, p)
        k1 = f(y)
        k2 = f(y + h * (1/5) * k1)
        k3 = f(y + h * ((3/40)*k1 + (9/40)*k2))
        k4 = f(y + h * ((44/45)*k1 - (56/15)*k2 + (32/9)*k3))
        k5 = f(y + h * ((19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4))
        k6 = f(y + h * ((9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5))
        y5 = y + h * ((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
        k7 = f(y5)
        y4 = y + h * ((5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error = float(np.max(np.abs(y5 - y4) / scale))
        if error <= 1.0:
            y = y5
            t += h
        factor = 5.0 if error == 0.0 else min(5.0, max(0.2, 0.9 * error ** (-0.2)))
        h = max(1e-6, h * factor)
    return np.clip(y, 0.0, 1.0)


def solver_crosscheck() -> pd.DataFrame:
    """Compare Euler output with an independently coded adaptive RK45 path."""
    rows = []
    for dwell in (50.0, 320.0):
        euler = simulate_general(dwell=dwell, n_levels=31, dt=0.2)
        up = np.linspace(0.0, 1.0, 31)
        down = np.linspace(1.0, 0.0, 31)[1:]
        schedule = [float(a) for a in up] + [float(a) for a in down]
        state = np.array([0.001, 0.0, 0.0, 0.0], dtype=float)
        rk_states = []
        for alpha in schedule:
            state = _rk45_constant_control(state, alpha, dwell, core.BASE)
            rk_states.append(state.copy())
        rk = np.vstack(rk_states)
        eu = euler[["G", "M", "L", "C"]].to_numpy(dtype=float)
        difference = np.abs(eu - rk)
        rows.append(
            {
                "dwell": dwell,
                "max_absolute_state_difference": float(np.max(difference)),
                "max_absolute_G_difference": float(np.max(difference[:, 0])),
                "endpoint_l2_difference": float(np.linalg.norm(eu[-1] - rk[-1])),
            }
        )
    return pd.DataFrame(rows)


def _full_system_multistable(p: core.Parameters) -> bool:
    for alpha in np.linspace(0.0, 1.0, 41):
        stable = 0
        for g in equilibrium_roots(float(alpha), p, 2001):
            state = np.r_[g, stationary_reservoirs(g, float(alpha), p)]
            if float(np.max(np.linalg.eigvals(full_jacobian(state, float(alpha), p)).real)) < -STABILITY_TOL:
                stable += 1
        if stable > 1:
            return True
    return False


def global_sensitivity(n_draws: int = 1000, seed: int = 20270826) -> pd.DataFrame:
    """Latin-hypercube audit over every free parameter except the time unit."""
    varied = [key for key in asdict(core.BASE) if key != "tau_g"]
    base = asdict(core.BASE)
    rng = np.random.default_rng(seed)
    unit = np.empty((n_draws, len(varied)), dtype=float)
    for column in range(len(varied)):
        unit[:, column] = (rng.permutation(n_draws) + rng.random(n_draws)) / n_draws
    multipliers = 0.8 + 0.4 * unit
    rows = []
    for draw, vector in enumerate(multipliers):
        values = dict(base)
        for key, multiplier in zip(varied, vector):
            values[key] = base[key] * float(multiplier)
        p = core.Parameters(**values)
        frame = simulate_general(p=p, dwell=25.0, n_levels=21, dt=0.25)
        result = metrics(frame, p=p, dt=0.25)
        up, down = core.aligned_branches(frame)
        up_cross = up[up.G >= 0.5]
        down_cross = down[down.G < 0.5]
        threshold_width = math.nan
        if not up_cross.empty and not down_cross.empty:
            threshold_width = float(up_cross.iloc[0].alpha - down_cross.iloc[-1].alpha)
        rows.append(
            {
                "draw": draw,
                **result,
                "threshold_width": threshold_width,
                "full_system_multistable": _full_system_multistable(p),
                **{f"multiplier_{key}": float(value) for key, value in zip(varied, vector)},
            }
        )
    return pd.DataFrame(rows)


def fold_brackets(equilibria: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in equilibria.groupby("regime", sort=False):
        counts = group.groupby("alpha").size().sort_index()
        alpha = counts.index.to_numpy(dtype=float)
        values = counts.to_numpy(dtype=int)
        for index in np.where(values[1:] != values[:-1])[0]:
            rows.append(
                {
                    "regime": regime,
                    "alpha_left": float(alpha[index]),
                    "alpha_right": float(alpha[index + 1]),
                    "roots_left": int(values[index]),
                    "roots_right": int(values[index + 1]),
                    "bracket_width": float(alpha[index + 1] - alpha[index]),
                }
            )
    return pd.DataFrame(rows)


def bifurcation_figure(path: Path, equilibria: pd.DataFrame) -> None:
    """Stable/unstable fixed-alpha branches with finite-rate trajectories."""
    width, height = 7.1 * inch, 3.15 * inch
    c = canvas.Canvas(str(path), pagesize=(width, height), initialFontName="FigureSans")
    panels = [
        ("fast-subsystem monostable + slow memory", "a"),
        ("bistable + memory", "b"),
    ]
    for index, (regime, label) in enumerate(panels):
        x0 = (0.62 + 3.43 * index) * inch
        y0, w, h = 0.63 * inch, 2.75 * inch, 1.94 * inch
        c.setFillColor(PALE); c.rect(x0, y0, w, h, fill=1, stroke=0)
        group = equilibria[equilibria.regime == regime]
        for stable, color, radius in ((False, GREY, 0.65), (True, ORANGE, 0.85)):
            points = group[group.stable == stable]
            c.setFillColor(color)
            for row in points.itertuples(index=False):
                c.circle(x0 + float(row.alpha) * w, y0 + float(row.G) * h, radius, fill=1, stroke=0)
        p = structural_regimes()[regime]
        dynamic = simulate_general(p=p, dwell=50.0, n_levels=31, dt=0.2)
        for branch, color in (("up", BLUE), ("down", GREEN)):
            branch_frame = dynamic[dynamic.branch == branch].sort_values("alpha")
            xs = x0 + branch_frame.alpha.to_numpy() * w
            ys = y0 + branch_frame.G.to_numpy() * h
            core.draw_polyline(c, xs, ys, color, 1.35, [4, 2] if branch == "down" else None)
        c.setStrokeColor(black); c.setLineWidth(0.7); c.rect(x0, y0, w, h, fill=0, stroke=1)
        _font(c, 6.3); c.setFillColor(black)
        for tick in (0.0, 0.5, 1.0):
            c.drawCentredString(x0 + tick * w, y0 - 11, f"{tick:.1f}")
            c.drawRightString(x0 - 4, y0 + tick * h - 2, f"{tick:.1f}")
        c.drawCentredString(x0 + w/2, y0 - 23, "fixed mediation alpha")
        c.saveState(); c.translate(x0 - 25, y0 + h/2); c.rotate(90); c.drawCentredString(0, 0, "equilibrium / dynamic G"); c.restoreState()
        _font(c, 7.0, True); c.drawString(x0 - 0.28*inch, y0 + h + 0.19*inch, label)
        _font(c, 6.1); c.drawString(x0, y0 + h + 0.13*inch, regime)
    _font(c, 6.0); c.setFillColor(ORANGE); c.drawString(1.55*inch, 2.92*inch, "stable equilibria")
    c.setFillColor(GREY); c.drawString(2.48*inch, 2.92*inch, "unstable equilibria")
    c.setFillColor(BLUE); c.drawString(4.53*inch, 2.92*inch, "up trajectory")
    c.setFillColor(GREEN); c.drawString(5.35*inch, 2.92*inch, "down trajectory")
    _font(c, 6.1); c.setFillColor(GREY)
    c.drawCentredString(width/2, 0.13*inch, "401-point alpha scan; tangential-root safeguard; dynamic dwell 50")
    c.showPage(); c.save()


def _font(c: canvas.Canvas, size: float, bold: bool = False) -> None:
    core.set_font(c, size, bold)


def figure(path: Path, regimes: pd.DataFrame, interventions: pd.DataFrame, noisy: pd.DataFrame) -> None:
    """Four-panel vector summary of discrimination and intervention semantics."""
    width, height = 7.1 * inch, 5.35 * inch
    c = canvas.Canvas(str(path), pagesize=(width, height), initialFontName="FigureSans")

    # Panel a: rate response across structural regimes.
    rect = (0.67 * inch, 3.08 * inch, 2.70 * inch, 1.62 * inch)
    tr = core.draw_axes(
        c, rect, (math.log10(5), math.log10(320)), (0, 0.52),
        "dwell per level (log)", "absolute path deficit",
        [math.log10(v) for v in (5, 20, 80, 320)], [0, .1, .2, .3, .4, .5],
        xformatter=lambda value: f"{int(round(10**value))}",
    )
    styles = [
        ("fast-subsystem monostable, no memory", GREY, [2, 2]),
        ("fast-subsystem monostable + slow memory", GREEN, None),
        ("near-critical fast subsystem + memory", PURPLE, None),
        ("bistable, no memory", BLUE, [5, 2]),
        ("bistable + memory", ORANGE, None),
    ]
    for label, color, dash in styles:
        group = regimes[(regimes.regime == label) & (regimes.link == "logistic")].sort_values("dwell")
        x, y = tr(np.log10(group.dwell), group.absolute_area)
        core.draw_polyline(c, x, y, color, 1.45, dash)
    core.panel_label(c, "a", 0.35 * inch, 4.84 * inch)

    # One shared row keeps all five mechanism classes clear of both x axes.
    legend_labels = [
        "full mono; no mem",
        "fast mono/full multi",
        "near crit/full multi",
        "full multi; no mem",
        "full multi + mem",
    ]
    _font(c, 6.0)
    for idx, ((_, color, dash), label) in enumerate(zip(styles, legend_labels)):
        xx = (0.26 + 1.38 * idx) * inch
        yy = 2.43 * inch
        c.setStrokeColor(color); c.setLineWidth(1.3); c.setDash(dash or [])
        c.line(xx, yy, xx + 0.20 * inch, yy); c.setDash([])
        c.setFillColor(black); c.drawString(xx + 0.24 * inch, yy - 2, label)

    # Panel b: alternative links under matched local slope.
    rect = (4.00 * inch, 3.08 * inch, 2.48 * inch, 1.62 * inch)
    tr = core.draw_axes(
        c, rect, (math.log10(20), math.log10(320)), (0, 0.52),
        "dwell per level (log)", "absolute path deficit",
        [math.log10(v) for v in (20, 80, 320)], [0, .1, .2, .3, .4, .5],
        xformatter=lambda value: f"{int(round(10**value))}",
    )
    for link, color in (("logistic", ORANGE), ("probit", BLUE)):
        label = f"bistable + memory ({link})"
        group = regimes[regimes.regime == label].sort_values("dwell")
        x, y = tr(np.log10(group.dwell), group.absolute_area)
        core.draw_polyline(c, x, y, color, 1.7)
        c.setFillColor(color)
        for xx, yy in zip(x, y): c.circle(float(xx), float(yy), 1.8, fill=1, stroke=0)
    core.draw_legend(c, [("logistic", ORANGE, None), ("probit", BLUE, None)], 4.30 * inch, 4.53 * inch, 14)
    core.panel_label(c, "b", 3.68 * inch, 4.84 * inch)

    # Panel c: ablation and reset are visibly different operations.
    labels = [("none", ""), ("ablate", "M+L"), ("one-shot", "M+L"),
              ("restore", "M+L"), ("C", "recovery"), ("restore", "+ recovery")]
    lookup = interventions.set_index("intervention")
    keys = ["none", "ablate M+L", "zero M+L once", "reference-restore M+L", "facilitate C recovery", "reference-restore M+L + C recovery"]
    x0, y0, w, h = 0.70 * inch, 0.66 * inch, 3.05 * inch, 1.48 * inch
    c.setFillColor(PALE); c.rect(x0, y0, w, h, fill=1, stroke=0)
    # Endpoint-state deficits can exceed the path-deficit scale; keep every bar
    # inside the frame rather than clipping the largest state-restoration gaps.
    ymax = 0.70
    bar_w = w / (len(keys) * 2.35)
    for i, (label_lines, key) in enumerate(zip(labels, keys)):
        xx = x0 + (i + .52) * w / len(keys)
        path_value = float(lookup.loc[key, "absolute_area"])
        state_value = float(lookup.loc[key, "endpoint_state_deficit"])
        for offset, value, color in ((-bar_w*.55, path_value, ORANGE), (bar_w*.55, state_value, BLUE)):
            bh = value / ymax * h
            c.setFillColor(color); c.rect(xx + offset - bar_w/2, y0, bar_w, bh, fill=1, stroke=0)
        _font(c, 6.2); c.setFillColor(black)
        c.drawCentredString(xx, y0 - 10, label_lines[0])
        if label_lines[1]:
            c.drawCentredString(xx, y0 - 18, label_lines[1])
    c.setStrokeColor(black); c.rect(x0, y0, w, h, fill=0, stroke=1)
    _font(c, 6.1); c.setFillColor(ORANGE); c.drawString(x0 + 0.05*inch, y0+h+7, "path")
    c.setFillColor(BLUE); c.drawString(x0 + 0.47*inch, y0+h+7, "state")
    core.panel_label(c, "c", 0.35 * inch, 2.27 * inch)

    # Panel d: stochastic stress distribution, rendered as quantile bars.
    x0, y0, w, h = 4.25 * inch, 0.66 * inch, 2.15 * inch, 1.48 * inch
    c.setFillColor(PALE); c.rect(x0, y0, w, h, fill=1, stroke=0)
    for tick in (0, .25, .50):
        yy = y0 + tick / .55 * h
        c.setStrokeColor(LIGHT); c.setLineWidth(.6); c.line(x0, yy, x0+w, yy)
        _font(c, 6.2); c.setFillColor(black); c.drawRightString(x0-4, yy-2, f"{tick:.2g}")
    for i, (link, color) in enumerate((("logistic", ORANGE), ("probit", BLUE))):
        values = noisy[noisy.link == link].absolute_area.to_numpy()
        q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
        xx = x0 + (i + .65) * w / 2
        scale = lambda value: y0 + value / 0.55 * h
        c.setStrokeColor(color); c.setLineWidth(1.4)
        c.line(xx, scale(q05), xx, scale(q95))
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=.20))
        c.rect(xx - .17*inch, scale(q25), .34*inch, scale(q75)-scale(q25), fill=1, stroke=1)
        c.setStrokeColor(color); c.setLineWidth(2.0); c.line(xx-.17*inch, scale(q50), xx+.17*inch, scale(q50))
        _font(c, 6.2); c.setFillColor(black); c.drawCentredString(xx, y0-12, link)
    c.setStrokeColor(black); c.rect(x0, y0, w, h, fill=0, stroke=1)
    _font(c, 6.4); c.saveState(); c.translate(x0-23, y0+h/2); c.rotate(90); c.drawCentredString(0,0,"absolute path deficit"); c.restoreState()
    core.panel_label(c, "d", 3.90 * inch, 2.27 * inch)
    _font(c, 6.2); c.setFillColor(GREY)
    c.drawCentredString(x0+w/2, 0.13*inch, "64 seeded process-noise runs per link")

    c.showPage(); c.save()


def main() -> None:
    for directory in (DATA, FIGURES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    regimes = structural_regime_sweep()
    equilibria, classifications = classify_full_system()
    refinement = refinement_audit()
    folds = fold_brackets(equilibria)
    interventions = intervention_comparison()
    noisy = stochastic_link_stress()
    solver_audit = solver_crosscheck()
    global_audit = global_sensitivity()
    regimes.to_csv(DATA / "model_i_structural_regimes.csv", index=False)
    equilibria.to_csv(DATA / "model_i_full_system_equilibria.csv", index=False)
    classifications.to_csv(DATA / "model_i_full_system_classification.csv", index=False)
    refinement.to_csv(DATA / "model_i_equilibrium_grid_refinement.csv", index=False)
    folds.to_csv(DATA / "model_i_fold_brackets.csv", index=False)
    interventions.to_csv(DATA / "model_i_interventions.csv", index=False)
    noisy.to_csv(DATA / "model_i_stochastic_links.csv", index=False)
    solver_audit.to_csv(DATA / "model_i_solver_crosscheck.csv", index=False)
    global_audit.to_csv(DATA / "model_i_global_sensitivity.csv", index=False)
    figure(FIGURES / "fig4_sensitivity.pdf", regimes, interventions, noisy)
    bifurcation_figure(FIGURES / "fig5_bifurcation.pdf", equilibria)
    summary = {
        "status": "synthetic structural stress test; not an empirical estimate",
        "base_model": asdict(core.BASE),
        "terminology": {
            "ablation": "reservoir contribution removed throughout the path",
            "zeroing": "reservoir state assigned to zero at reversal; diagnostic only",
            "restoration": "assignment to and tracking of a contemporaneous alpha=0 reference trajectory",
            "human_recovery": "increased recovery rate; never literal erasure",
        },
        "regime_results": regimes.to_dict(orient="records"),
        "full_system_classification": classifications.to_dict(orient="records"),
        "equilibrium_grid_refinement": refinement.to_dict(orient="records"),
        "fold_brackets": folds.to_dict(orient="records"),
        "intervention_results": interventions.to_dict(orient="records"),
        "solver_crosscheck": solver_audit.to_dict(orient="records"),
        "global_sensitivity": {
            "n_draws": int(len(global_audit)),
            "seed": 20270826,
            "parameters_varied": [key for key in asdict(core.BASE) if key != "tau_g"],
            "absolute_area_quantiles": {
                str(q): float(global_audit.absolute_area.quantile(q)) for q in (.05, .5, .95)
            },
            "endpoint_state_deficit_quantiles": {
                str(q): float(global_audit.endpoint_state_deficit.quantile(q)) for q in (.05, .5, .95)
            },
            "full_system_multistable_fraction": float(global_audit.full_system_multistable.mean()),
        },
        "numerical_definitions": {
            "path_area": "composite trapezoidal rule over matched alpha, divided by alpha_max",
            "state_deficit": "Euclidean distance in (G,M,L,C) from same-elapsed alpha=0 reference, divided by 2",
            "process_noise": "independent Gaussian increments applied only to G after each Euler drift step; sd=0.012*sqrt(dt); all states clipped to [0,1]",
            "process_noise_seeds": [270000, 270127],
            "common_random_numbers": "the same seed and Gaussian stream are used for the paired logistic and probit link runs",
            "probit_slope_scale": PROBIT_SLOPE_SCALE,
            "probit_matching": "Phi(c*k*field), c=sqrt(2*pi)/4, matching the logistic derivative k/4 at field zero",
            "bisection_iterations": BISECTION_ITERATIONS,
            "tangency_residual_tolerance": TANGENCY_RESIDUAL_TOL,
            "near_grid_residual_tolerance": GRID_RESIDUAL_TOL,
            "root_deduplication_tolerance": ROOT_DEDUP_TOL,
            "stability_tolerance": STABILITY_TOL,
            "human_recovery_multiplier": RECOVERY_MULTIPLIER,
        },
        "stochastic_link_quantiles": {
            link: {
                str(q): float(noisy[noisy.link == link].absolute_area.quantile(q))
                for q in (.05, .5, .95)
            }
            for link in ("logistic", "probit")
        },
    }
    (RESULTS / "model_i_stress_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "rows": {"regimes": len(regimes), "equilibria": len(equilibria), "classifications": len(classifications), "refinement": len(refinement), "folds": len(folds), "interventions": len(interventions), "noise": len(noisy), "global_sensitivity": len(global_audit)}}, indent=2))


if __name__ == "__main__":
    main()
