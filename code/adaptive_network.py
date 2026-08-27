"""Deterministic adaptive-network demonstration of algorithmic path dependence.

This is a synthetic mechanism demonstration, not an empirical fit.  A binary,
undirected follow graph coevolves with continuous node opinions while the share
``alpha`` of recommendation-ranked exposure follows an up-down protocol.

The implementation deliberately uses only NumPy, pandas, and ReportLab, which
are already pinned by the repository.  No graph library is required: all four
network observables are computed directly from the adjacency matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import reportlab
from reportlab.lib.colors import Color, HexColor, black
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

BLUE = HexColor("#1769AA")
ORANGE = HexColor("#D86F1D")
GREEN = HexColor("#238463")
GREY = HexColor("#626A73")
LIGHT_GREY = HexColor("#D7DDE4")
PALE = HexColor("#F4F6F8")

_REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("AdaptiveSans", _REPORTLAB_FONTS / "Vera.ttf"))
pdfmetrics.registerFont(TTFont("AdaptiveSans-Bold", _REPORTLAB_FONTS / "VeraBd.ttf"))


@dataclass(frozen=True)
class NetworkParameters:
    """Transparent parameters for the illustrative adaptive network."""

    n_nodes: int = 80
    n_edges: int = 240
    initial_bridge_fraction: float = 0.40
    n_levels: int = 21
    baseline_dwell: int = 20
    slow_dwell: int = 80
    burn_in: int = 120
    initial_opinion_anchor: float = 0.16
    initial_opinion_sd: float = 0.022
    opinion_step: float = 0.13
    identity_anchor: float = 0.18
    chronological_social_weight: float = 0.82
    ranked_extremity: float = 0.94
    ranked_self_weight: float = 2.4
    ranked_identity_weight: float = 0.72
    rewire_onset: float = 0.50
    rewire_rate: float = 0.012
    reset_alpha: float = 0.50
    seed: int = 20270824


BASE = NetworkParameters()
METRICS = ("Q", "r", "phi", "B")


def initialise_system(p: NetworkParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a connected graph, opinions, fixed camps, and deterministic tie keys."""
    rng = np.random.default_rng(p.seed)
    n = p.n_nodes
    if n % 2:
        raise ValueError("n_nodes must be even for the balanced two-camp construction")
    max_edges = n * (n - 1) // 2
    if not n <= p.n_edges <= max_edges:
        raise ValueError("n_edges must permit a connected simple graph")

    camps = np.r_[np.full(n // 2, -1, dtype=int), np.full(n // 2, 1, dtype=int)]
    camps = camps[rng.permutation(n)]
    order = rng.permutation(n)
    adjacency = np.zeros((n, n), dtype=np.int8)

    def add_edge(i: int, j: int) -> bool:
        if i == j or adjacency[i, j]:
            return False
        adjacency[i, j] = adjacency[j, i] = 1
        return True

    # A Hamiltonian cycle guarantees connectedness before controlled mixing.
    for index, i in enumerate(order):
        add_edge(int(i), int(order[(index + 1) % n]))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if not adjacency[i, j]]
    cross = [pair for pair in pairs if camps[pair[0]] != camps[pair[1]]]
    within = [pair for pair in pairs if camps[pair[0]] == camps[pair[1]]]
    rng.shuffle(cross)
    rng.shuffle(within)
    target_cross = int(round(p.initial_bridge_fraction * p.n_edges))
    current_cross = sum(
        int(camps[i] != camps[j])
        for i in range(n)
        for j in range(i + 1, n)
        if adjacency[i, j]
    )
    for i, j in cross:
        if current_cross >= target_cross or int(adjacency.sum() // 2) >= p.n_edges:
            break
        if add_edge(i, j):
            current_cross += 1
    if current_cross != target_cross:
        raise RuntimeError("could not construct the exact initial cross-edge target")
    # Once the exact cross-edge target is reached, complete the edge budget only
    # with within-camp nonedges so the final bridge share cannot drift upward.
    for i, j in within:
        if int(adjacency.sum() // 2) >= p.n_edges:
            break
        add_edge(i, j)
    if int(adjacency.sum() // 2) != p.n_edges:
        raise RuntimeError("insufficient admissible within-camp nonedges")
    final_cross = sum(
        int(camps[i] != camps[j])
        for i in range(n)
        for j in range(i + 1, n)
        if adjacency[i, j]
    )
    if final_cross != target_cross:
        raise AssertionError("initializer changed the exact cross-edge target")

    noise = rng.normal(0.0, p.initial_opinion_sd, n)
    opinions = np.clip(
        p.initial_opinion_anchor * camps + noise - noise.mean(), -1.0, 1.0
    )
    tie_keys = rng.random((n, n))
    tie_keys = (tie_keys + tie_keys.T) / 2.0
    np.fill_diagonal(tie_keys, 0.0)

    # Pre-equilibrate the measured initial condition under chronological exposure.
    for _ in range(p.burn_in):
        opinions = update_opinions(adjacency, opinions, camps, alpha=0.0, p=p)
    return adjacency, opinions, camps, tie_keys


def update_opinions(
    adjacency: np.ndarray,
    opinions: np.ndarray,
    camps: np.ndarray,
    alpha: float,
    p: NetworkParameters,
) -> np.ndarray:
    """One synchronous bounded-confidence-free opinion update."""
    degree = adjacency.sum(axis=1).astype(float)
    if np.any(degree == 0):
        raise RuntimeError("the degree-preserving swap should never isolate a node")
    neighbour_mean = adjacency @ opinions / degree
    chronological_target = (
        p.chronological_social_weight * neighbour_mean
        + (1.0 - p.chronological_social_weight) * p.identity_anchor * camps
    )
    ranked_target = p.ranked_extremity * np.tanh(
        p.ranked_self_weight * opinions + p.ranked_identity_weight * camps
    )
    target = (1.0 - alpha) * chronological_target + alpha * ranked_target
    updated = opinions + p.opinion_step * (target - opinions)
    return np.clip(updated, -1.0, 1.0)


def homophilic_edge_swap(
    adjacency: np.ndarray,
    opinions: np.ndarray,
    camps: np.ndarray,
    tie_keys: np.ndarray,
) -> bool:
    """Replace two cross-camp edges by two within-camp edges, preserving every degree.

    Candidate bridge pairs are ordered by current opinion disagreement; the fixed
    tie-key matrix makes all residual choices reproducible.  No random draw occurs
    during a protocol run.
    """
    n = len(camps)
    bridges: list[tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] and camps[i] != camps[j]:
                left, right = (i, j) if camps[i] < camps[j] else (j, i)
                bridges.append(
                    (left, right, abs(float(opinions[i] - opinions[j])), float(tie_keys[i, j]))
                )
    bridges.sort(key=lambda edge: (-edge[2], edge[3], edge[0], edge[1]))
    # Searching the leading bridges is deterministic and sufficient at this size.
    for first_index, (a, b, _, _) in enumerate(bridges[:64]):
        for c, d, _, _ in bridges[first_index + 1 : 64]:
            if a == c or b == d:
                continue
            if adjacency[a, c] or adjacency[b, d]:
                continue
            adjacency[a, b] = adjacency[b, a] = 0
            adjacency[c, d] = adjacency[d, c] = 0
            adjacency[a, c] = adjacency[c, a] = 1
            adjacency[b, d] = adjacency[d, b] = 1
            return True
    return False


def edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.where(np.triu(adjacency, 1) == 1)
    return {(int(i), int(j)) for i, j in zip(rows, cols)}


def validate_adjacency(adjacency: np.ndarray) -> None:
    """Fail fast if the evolving graph ceases to be simple and undirected."""
    if not np.array_equal(adjacency, adjacency.T):
        raise AssertionError("A is not symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise AssertionError("A contains a self-loop")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise AssertionError("A is not binary")


def network_metrics(
    adjacency: np.ndarray,
    opinions: np.ndarray,
    camps: np.ndarray,
    initial_bridges: set[tuple[int, int]],
) -> dict[str, float]:
    """Compute modularity, opinion assortativity, conductance, and bridge persistence.

    Q is Newman-Girvan modularity for the fixed balanced camp partition.  r is the
    Pearson correlation of endpoint opinions after duplicating each undirected edge
    in both orientations.  phi is the cross-camp cut divided by the smaller camp
    volume.  B is the fraction of time-zero cross-camp edges that remain present.
    """
    n = len(camps)
    degree = adjacency.sum(axis=1).astype(float)
    m = float(degree.sum() / 2.0)
    if m <= 0:
        raise RuntimeError("network has no edges")

    q = 0.0
    for camp in (-1, 1):
        nodes = np.where(camps == camp)[0]
        internal = float(adjacency[np.ix_(nodes, nodes)].sum() / 2.0)
        volume = float(degree[nodes].sum())
        q += internal / m - (volume / (2.0 * m)) ** 2

    rows, cols = np.where(np.triu(adjacency, 1) == 1)
    endpoint_a = np.r_[opinions[rows], opinions[cols]]
    endpoint_b = np.r_[opinions[cols], opinions[rows]]
    if float(endpoint_a.std()) < 1e-12 or float(endpoint_b.std()) < 1e-12:
        assortativity = 0.0
    else:
        assortativity = float(np.corrcoef(endpoint_a, endpoint_b)[0, 1])

    cross_mask = camps[rows] != camps[cols]
    cut = float(cross_mask.sum())
    volumes = [float(degree[camps == camp].sum()) for camp in (-1, 1)]
    conductance = cut / min(volumes)

    current = edge_set(adjacency)
    bridge_persistence = (
        len(current.intersection(initial_bridges)) / len(initial_bridges)
        if initial_bridges
        else 1.0
    )
    return {
        "Q": float(q),
        "r": assortativity,
        "phi": float(conductance),
        "B": float(bridge_persistence),
        "polarization": float(np.mean(np.abs(opinions))),
        "mean_opinion": float(np.mean(opinions)),
        "n_edges": int(m),
        "n_bridges": int(cut),
    }


def simulate_protocol(
    p: NetworkParameters,
    condition: str,
    dwell: int,
    structural_reset: bool,
) -> pd.DataFrame:
    """Simulate one up-down alpha path from an identical measured initial state."""
    initial_adjacency, initial_opinions, camps, tie_keys = initialise_system(p)
    adjacency = initial_adjacency.copy()
    opinions = initial_opinions.copy()
    initial_edges = edge_set(initial_adjacency)
    initial_bridges = {
        (i, j) for i, j in initial_edges if camps[i] != camps[j]
    }
    initial_degree = initial_adjacency.sum(axis=1).copy()
    up = np.linspace(0.0, 1.0, p.n_levels)
    down = np.linspace(1.0, 0.0, p.n_levels)[1:]
    schedule = [("up", float(alpha)) for alpha in up] + [
        ("down", float(alpha)) for alpha in down
    ]
    rows: list[dict[str, float | int | str | bool]] = []
    rewire_budget = 0.0
    reset_done = False

    for level, (branch, alpha) in enumerate(schedule):
        reset_applied = False
        if structural_reset and branch == "down" and alpha <= p.reset_alpha and not reset_done:
            # This is a topology-only intervention: opinions are intentionally retained.
            adjacency = initial_adjacency.copy()
            rewire_budget = 0.0
            reset_done = True
            reset_applied = True

        for _ in range(dwell):
            opinions = update_opinions(adjacency, opinions, camps, alpha, p)
            intensity = max(0.0, alpha - p.rewire_onset) / (1.0 - p.rewire_onset)
            rewire_budget += p.rewire_rate * p.n_nodes * intensity
            while rewire_budget >= 1.0:
                if not homophilic_edge_swap(adjacency, opinions, camps, tie_keys):
                    rewire_budget = 0.0
                    break
                rewire_budget -= 1.0

        if not np.array_equal(adjacency.sum(axis=1), initial_degree):
            raise AssertionError("edge swap changed the degree sequence")
        validate_adjacency(adjacency)
        metrics = network_metrics(adjacency, opinions, camps, initial_bridges)
        rows.append(
            {
                "condition": condition,
                "level": level,
                "branch": branch,
                "alpha": alpha,
                "dwell": dwell,
                "structural_reset": structural_reset,
                "reset_applied_here": reset_applied,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def aligned_branches(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    up = frame[frame.branch == "up"].sort_values("alpha").reset_index(drop=True)
    peak = up.iloc[[-1]].copy()
    peak.loc[:, "branch"] = "down"
    down = pd.concat([frame[frame.branch == "down"], peak], ignore_index=True)
    down = down.sort_values("alpha").reset_index(drop=True)
    if not np.allclose(up.alpha, down.alpha):
        raise AssertionError("up and down branches do not share the same alpha grid")
    return up, down


def branch_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    up, down = aligned_branches(frame)
    rows = {
        "condition": frame.condition.iloc[0],
        "alpha": up.alpha.to_numpy(),
    }
    for metric in METRICS:
        rows[f"delta_{metric}"] = down[metric].to_numpy() - up[metric].to_numpy()
    return pd.DataFrame(rows)


def loop_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    gaps = branch_gaps(frame)
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = gaps[f"delta_{metric}"].to_numpy()
        alpha = gaps.alpha.to_numpy()
        summary[metric] = {
            "signed_area_down_minus_up": float(np.trapezoid(values, alpha)),
            "absolute_area": float(np.trapezoid(np.abs(values), alpha)),
            "max_absolute_gap": float(np.max(np.abs(values))),
            "gap_at_alpha_0": float(values[0]),
        }
    return summary


def set_font(c: canvas.Canvas, size: float, bold: bool = False) -> None:
    c.setFont("AdaptiveSans-Bold" if bold else "AdaptiveSans", size)


def draw_polyline(
    c: canvas.Canvas,
    xs: np.ndarray,
    ys: np.ndarray,
    color: Color,
    width: float,
    dash: list[float] | None = None,
) -> None:
    path = c.beginPath()
    for index, (x, y) in enumerate(zip(xs, ys)):
        if index == 0:
            path.moveTo(float(x), float(y))
        else:
            path.lineTo(float(x), float(y))
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.drawPath(path, stroke=1, fill=0)
    c.setDash([])


def figure_adaptive_network(path: Path, gaps: pd.DataFrame, p: NetworkParameters) -> None:
    """Create a compact one-column vector figure of matched-current branch gaps."""
    width, height = 3.45 * inch, 5.35 * inch
    c = canvas.Canvas(str(path), pagesize=(width, height), initialFontName="AdaptiveSans")
    left = 0.52 * inch
    right = width - 0.15 * inch
    plot_width = right - left
    top = height - 0.60 * inch
    panel_height = 0.86 * inch
    panel_gap = 0.23 * inch
    labels = {
        "Q": "modularity  delta Q",
        "r": "opinion assort.  delta r",
        "phi": "conductance  delta phi",
        "B": "bridge persist.  delta B",
    }
    styles = [
        ("baseline", ORANGE, None),
        ("slow", BLUE, [4, 2]),
        ("topology reset", GREEN, [1.2, 1.8]),
    ]

    # Legend occupies its own band to prevent collisions with panel labels.
    legend_y = height - 0.30 * inch
    legend_x = 0.16 * inch
    set_font(c, 6.45)
    for label, color, dash in styles:
        c.setStrokeColor(color)
        c.setLineWidth(1.5)
        c.setDash(dash or [])
        c.line(legend_x, legend_y, legend_x + 0.22 * inch, legend_y)
        c.setDash([])
        c.setFillColor(black)
        c.drawString(legend_x + 0.26 * inch, legend_y - 2.2, label)
        legend_x += {"baseline": 0.91, "slow": 0.70, "topology reset": 1.28}[label] * inch

    for panel_index, metric in enumerate(METRICS):
        y0 = top - panel_index * (panel_height + panel_gap) - panel_height
        metric_values = gaps[f"delta_{metric}"].to_numpy()
        span = max(0.015, float(np.max(np.abs(metric_values))) * 1.16)
        # Symmetric axes make direction and reset attenuation immediately legible.
        y_min, y_max = -span, span
        c.setFillColor(PALE)
        c.rect(left, y0, plot_width, panel_height, fill=1, stroke=0)
        c.setStrokeColor(LIGHT_GREY)
        c.setLineWidth(0.45)
        zero_y = y0 + (-y_min) / (y_max - y_min) * panel_height
        c.line(left, zero_y, right, zero_y)
        for tick in (0.0, 0.5, 1.0):
            x = left + tick * plot_width
            c.setStrokeColor(Color(0.84, 0.87, 0.90, alpha=0.7))
            c.line(x, y0, x, y0 + panel_height)

        c.setStrokeColor(black)
        c.setLineWidth(0.65)
        c.rect(left, y0, plot_width, panel_height, fill=0, stroke=1)
        set_font(c, 6.35)
        c.setFillColor(black)
        c.drawRightString(left - 4, y0 - 1, f"{y_min:.2f}")
        c.drawRightString(left - 4, y0 + panel_height - 2, f"{y_max:.2f}")
        c.drawRightString(left - 4, zero_y - 2, "0")
        # Keep the descriptor in the inter-panel band rather than over a curve.
        set_font(c, 6.8, True)
        c.drawString(left, y0 + panel_height + 4, f"{chr(97 + panel_index)}   {labels[metric]}")

        for condition, color, dash in styles:
            group = gaps[gaps.condition == condition].sort_values("alpha")
            xs = left + group.alpha.to_numpy() * plot_width
            values = group[f"delta_{metric}"].to_numpy()
            ys = y0 + (values - y_min) / (y_max - y_min) * panel_height
            draw_polyline(c, xs, ys, color, 1.35, dash)

        if panel_index == len(METRICS) - 1:
            set_font(c, 6.4)
            for tick in (0.0, 0.5, 1.0):
                x = left + tick * plot_width
                c.drawCentredString(x, y0 - 11, f"{tick:.1f}")
            set_font(c, 7.0)
            c.drawCentredString((left + right) / 2, y0 - 23, "current mediation  alpha")

    set_font(c, 5.9)
    c.setFillColor(GREY)
    c.drawCentredString(
        width / 2,
        0.08 * inch,
        f"synthetic; down - up at matched alpha; seed {p.seed}; reset at alpha={p.reset_alpha:.2f}",
    )
    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter dwell times for a fast smoke test; outputs remain synthetic.",
    )
    args = parser.parse_args()
    p = BASE
    baseline_dwell = 8 if args.quick else p.baseline_dwell
    slow_dwell = 24 if args.quick else p.slow_dwell
    for directory in (DATA, FIGURES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)

    runs = [
        simulate_protocol(p, "baseline", baseline_dwell, structural_reset=False),
        simulate_protocol(p, "slow", slow_dwell, structural_reset=False),
        simulate_protocol(p, "topology reset", baseline_dwell, structural_reset=True),
    ]
    trajectories = pd.concat(runs, ignore_index=True)
    gaps = pd.concat([branch_gaps(run) for run in runs], ignore_index=True)
    trajectories.to_csv(DATA / "adaptive_network_trajectories.csv", index=False)
    gaps.to_csv(DATA / "adaptive_network_branch_gaps.csv", index=False)

    initial_row = runs[0].iloc[0]
    summaries = {run.condition.iloc[0]: loop_summary(run) for run in runs}
    baseline_abs = {metric: summaries["baseline"][metric]["absolute_area"] for metric in METRICS}
    reset_abs = {metric: summaries["topology reset"][metric]["absolute_area"] for metric in METRICS}
    summary = {
        "status": "synthetic mechanism demonstration; not an empirical estimate",
        "model": asdict(p),
        "protocol": {
            "path": "0 -> 1 -> 0",
            "ascending_dwell_blocks": p.n_levels,
            "descending_dwell_blocks": p.n_levels - 1,
            "turning_point_counted_once": True,
            "matched_branch_grid_points": p.n_levels,
            "baseline_dwell": baseline_dwell,
            "slow_dwell": slow_dwell,
            "topology_reset": (
                "On the descending branch at alpha=0.50, restore A(0) only; retain x."
            ),
        },
        "metric_definitions": {
            "Q": "Newman-Girvan modularity of the fixed two-camp partition.",
            "r": "Pearson correlation of node opinions across oriented edge endpoints.",
            "phi": "Cross-camp cut size divided by the smaller camp volume.",
            "B": "Fraction of time-zero cross-camp edges still present.",
        },
        "initial_observables": {metric: float(initial_row[metric]) for metric in METRICS},
        "loop_metrics": summaries,
        "reset_absolute_area_reduction_fraction": {
            metric: float(1.0 - reset_abs[metric] / baseline_abs[metric])
            if baseline_abs[metric] > 0
            else 0.0
            for metric in METRICS
        },
        "invariants_checked": [
            "A is symmetric and binary with zero diagonal.",
            "Every homophilic swap preserves the complete degree sequence.",
            "All conditions start from the identical graph and opinion state.",
        ],
    }
    (RESULTS / "adaptive_network_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    figure_adaptive_network(FIGURES / "fig6_adaptive_network.pdf", gaps, p)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
