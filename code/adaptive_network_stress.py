"""Reversible stochastic adaptive-network stress test across many seeds.

Unlike the legacy one-way illustration, this model permits degree-preserving
cross-to-within and within-to-cross rewiring.  It compares null, rapidly
reversible, slowly reversible, and one-way boundary regimes and reports
distributions across independent network initializations.  Results are
mechanism demonstrations, not platform estimates.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black
from reportlab.lib.pagesizes import inch
from reportlab.pdfgen import canvas

import adaptive_network as legacy


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

METRICS = ("Q", "r_x", "phi", "P")


@dataclass(frozen=True)
class Regime:
    name: str
    homophily_rate: float
    reverse_rate: float
    dwell: int
    graph_reset: bool = False


REGIMES = (
    Regime("no rewiring", 0.0, 0.0, 30),
    Regime("fast reversible", 0.010, 0.020, 30),
    Regime("fast reversible, slow ramp", 0.010, 0.020, 120),
    Regime("slow reversible", 0.010, 0.0025, 30),
    Regime("slow reversible, slow ramp", 0.010, 0.0025, 120),
    Regime("one-way boundary", 0.010, 0.0, 30),
    Regime("slow reversible + graph reset", 0.010, 0.0025, 30, True),
)


def oriented_cross_edges(adjacency: np.ndarray, camps: np.ndarray) -> list[tuple[int, int]]:
    edges = []
    rows, cols = np.where(np.triu(adjacency, 1) == 1)
    for i, j in zip(rows, cols):
        if camps[i] != camps[j]:
            edges.append((int(i), int(j)) if camps[i] < camps[j] else (int(j), int(i)))
    return edges


def stochastic_homophilic_swap(
    adjacency: np.ndarray, camps: np.ndarray, rng: np.random.Generator
) -> bool:
    """Replace two cross-camp edges with within-camp edges, preserving degrees."""
    bridges = oriented_cross_edges(adjacency, camps)
    if len(bridges) < 2:
        return False
    for _ in range(96):
        first, second = rng.choice(len(bridges), size=2, replace=False)
        a, b = bridges[int(first)]
        c, d = bridges[int(second)]
        if len({a, b, c, d}) < 4 or adjacency[a, c] or adjacency[b, d]:
            continue
        adjacency[a, b] = adjacency[b, a] = 0
        adjacency[c, d] = adjacency[d, c] = 0
        adjacency[a, c] = adjacency[c, a] = 1
        adjacency[b, d] = adjacency[d, b] = 1
        return True
    return False


def stochastic_reverse_swap(
    adjacency: np.ndarray, camps: np.ndarray, rng: np.random.Generator
) -> bool:
    """Replace one within-camp edge per camp with two cross-camp edges."""
    rows, cols = np.where(np.triu(adjacency, 1) == 1)
    left = [(int(i), int(j)) for i, j in zip(rows, cols) if camps[i] == camps[j] == -1]
    right = [(int(i), int(j)) for i, j in zip(rows, cols) if camps[i] == camps[j] == 1]
    if not left or not right:
        return False
    for _ in range(128):
        a, c = left[int(rng.integers(len(left)))]
        b, d = right[int(rng.integers(len(right)))]
        candidates = ((a, b, c, d), (a, d, c, b))
        if rng.random() < 0.5:
            candidates = candidates[::-1]
        for u, v, x, y in candidates:
            if not adjacency[u, v] and not adjacency[x, y]:
                adjacency[a, c] = adjacency[c, a] = 0
                adjacency[b, d] = adjacency[d, b] = 0
                adjacency[u, v] = adjacency[v, u] = 1
                adjacency[x, y] = adjacency[y, x] = 1
                return True
    return False


def cross_count(adjacency: np.ndarray, camps: np.ndarray) -> int:
    rows, cols = np.where(np.triu(adjacency, 1) == 1)
    return int(np.sum(camps[rows] != camps[cols]))


def simulate(seed: int, regime: Regime) -> pd.DataFrame:
    p = replace(legacy.BASE, seed=seed, n_levels=21)
    initial_a, initial_x, camps, _ = legacy.initialise_system(p)
    adjacency = initial_a.copy()
    opinions = initial_x.copy()
    initial_degree = initial_a.sum(axis=1).copy()
    initial_bridges = {
        edge for edge in legacy.edge_set(initial_a) if camps[edge[0]] != camps[edge[1]]
    }
    target_cross = len(initial_bridges)
    # Opinion innovations use a dedicated common-random-number stream across
    # regimes within a seed. Rewiring proposals use a separate stream so a
    # different number of swaps cannot shift subsequent opinion innovations.
    opinion_rng = np.random.default_rng(seed + 91173)
    rewire_rng = np.random.default_rng(seed + 28183)
    up = np.linspace(0.0, 1.0, p.n_levels)
    down = np.linspace(1.0, 0.0, p.n_levels)[1:]
    schedule = [("up", float(a)) for a in up] + [("down", float(a)) for a in down]
    reset_done = False
    rows = []
    for level, (branch, alpha) in enumerate(schedule):
        reset_here = False
        if regime.graph_reset and branch == "down" and alpha <= p.reset_alpha and not reset_done:
            adjacency = initial_a.copy()
            reset_done = True
            reset_here = True
        cross_before = cross_count(adjacency, camps)
        hom_attempts = hom_accepted = reverse_attempts = reverse_accepted = 0
        for _ in range(regime.dwell):
            opinions = legacy.update_opinions(adjacency, opinions, camps, alpha, p)
            opinions = np.clip(
                opinions + opinion_rng.normal(0.0, 0.0035, len(opinions)), -1.0, 1.0
            )
            hom_intensity = max(0.0, alpha - p.rewire_onset) / (1.0 - p.rewire_onset)
            reverse_intensity = max(0.0, p.rewire_onset - alpha) / p.rewire_onset
            n_hom = int(rewire_rng.poisson(regime.homophily_rate * p.n_nodes * hom_intensity))
            n_reverse = int(rewire_rng.poisson(regime.reverse_rate * p.n_nodes * reverse_intensity))
            hom_attempts += n_hom
            reverse_attempts += n_reverse
            for _ in range(n_hom):
                if not stochastic_homophilic_swap(adjacency, camps, rewire_rng):
                    break
                hom_accepted += 1
            for _ in range(n_reverse):
                if cross_count(adjacency, camps) >= target_cross:
                    break
                if not stochastic_reverse_swap(adjacency, camps, rewire_rng):
                    break
                reverse_accepted += 1
        legacy.validate_adjacency(adjacency)
        if not np.array_equal(adjacency.sum(axis=1), initial_degree):
            raise AssertionError("stochastic swap changed the degree sequence")
        measured = legacy.network_metrics(adjacency, opinions, camps, initial_bridges)
        measured["r_x"] = measured.pop("r")
        measured["P"] = measured.pop("polarization")
        cross_after = int(measured["n_bridges"])
        predicted_per_update = 2.0 * p.n_nodes * (
            regime.reverse_rate * max(0.0, p.rewire_onset - alpha) / p.rewire_onset
            - regime.homophily_rate * max(0.0, alpha - p.rewire_onset) / (1.0 - p.rewire_onset)
        )
        rows.append(
            {
                "seed": seed,
                "regime": regime.name,
                "branch": branch,
                "level": level,
                "alpha": alpha,
                "dwell": regime.dwell,
                "homophily_rate": regime.homophily_rate,
                "reverse_rate": regime.reverse_rate,
                "graph_reset": regime.graph_reset,
                "reset_applied_here": reset_here,
                "cross_edge_fraction_of_initial": measured["n_bridges"] / target_cross,
                "cross_edges_before_rewiring_block": cross_before,
                "cross_edges_after_rewiring_block": cross_after,
                "observed_cross_edge_drift_per_update": (cross_after - cross_before) / regime.dwell,
                "interior_predicted_drift_per_update": predicted_per_update,
                "homophilic_attempts": hom_attempts,
                "homophilic_accepted": hom_accepted,
                "reverse_attempts": reverse_attempts,
                "reverse_accepted": reverse_accepted,
                "homophilic_acceptance_rate": hom_accepted / hom_attempts if hom_attempts else np.nan,
                "reverse_acceptance_rate": reverse_accepted / reverse_attempts if reverse_attempts else np.nan,
                **measured,
            }
        )
    return pd.DataFrame(rows)


def aligned(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    up = frame[frame.branch == "up"].sort_values("alpha").reset_index(drop=True)
    peak = up.iloc[[-1]].copy(); peak.loc[:, "branch"] = "down"
    down = pd.concat([frame[frame.branch == "down"], peak], ignore_index=True)
    down = down.sort_values("alpha").reset_index(drop=True)
    return up, down


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    up, down = aligned(frame)
    gap_rows = {"seed": frame.seed.iloc[0], "regime": frame.regime.iloc[0], "alpha": up.alpha.to_numpy()}
    summaries = []
    for metric in METRICS:
        gap = down[metric].to_numpy() - up[metric].to_numpy()
        gap_rows[f"delta_{metric}"] = gap
        summaries.append(
            {
                "seed": int(frame.seed.iloc[0]),
                "regime": frame.regime.iloc[0],
                "metric": metric,
                "signed_area": float(np.trapezoid(gap, up.alpha.to_numpy())),
                "absolute_area": float(np.trapezoid(np.abs(gap), up.alpha.to_numpy())),
                "max_absolute_gap": float(np.max(np.abs(gap))),
                "gap_at_alpha_0": float(gap[0]),
            }
        )
    return pd.DataFrame(gap_rows), pd.DataFrame(summaries)


def quantile_summary(summaries: pd.DataFrame) -> list[dict]:
    rows = []
    for (regime, metric), group in summaries.groupby(["regime", "metric"], sort=False):
        for quantity in ("absolute_area", "gap_at_alpha_0"):
            q = group[quantity].quantile([.05, .25, .5, .75, .95])
            rows.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "quantity": quantity,
                    **{f"q{int(level*100):02d}": float(value) for level, value in q.items()},
                }
            )
    return rows


def drift_validation(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed drift and swap acceptance by regime and alpha."""
    rows = []
    for (regime, alpha), group in trajectories.groupby(["regime", "alpha"], sort=False):
        hom_attempts = int(group.homophilic_attempts.sum())
        reverse_attempts = int(group.reverse_attempts.sum())
        rows.append(
            {
                "regime": regime,
                "alpha": float(alpha),
                "mean_observed_drift_per_update": float(group.observed_cross_edge_drift_per_update.mean()),
                "interior_predicted_drift_per_update": float(group.interior_predicted_drift_per_update.iloc[0]),
                "homophilic_attempts": hom_attempts,
                "homophilic_accepted": int(group.homophilic_accepted.sum()),
                "homophilic_acceptance_rate": float(group.homophilic_accepted.sum() / hom_attempts) if hom_attempts else math.nan,
                "reverse_attempts": reverse_attempts,
                "reverse_accepted": int(group.reverse_accepted.sum()),
                "reverse_acceptance_rate": float(group.reverse_accepted.sum() / reverse_attempts) if reverse_attempts else math.nan,
            }
        )
    return pd.DataFrame(rows)


def drift_figure(path: Path, validation: pd.DataFrame) -> None:
    width, height = 7.1 * inch, 3.05 * inch
    c = canvas.Canvas(str(path), pagesize=(width, height), initialFontName="AdaptiveSans")
    regimes = (("fast reversible", BLUE), ("slow reversible", ORANGE), ("one-way boundary", PURPLE))
    for panel, (quantity, y_min, y_max, title) in enumerate((
        ("drift", -1.8, 3.4, "cross-edge drift per update"),
        ("acceptance", 0.0, 1.05, "proposal acceptance probability"),
    )):
        x0 = (0.66 + 3.46 * panel) * inch
        y0, w, h = 0.63 * inch, 2.72 * inch, 1.78 * inch
        c.setFillColor(PALE); c.rect(x0, y0, w, h, fill=1, stroke=0)
        zero = y0 + (0.0 - y_min) / (y_max - y_min) * h
        c.setStrokeColor(LIGHT); c.line(x0, zero, x0+w, zero)
        for regime, color in regimes:
            group = validation[validation.regime == regime].sort_values("alpha")
            xs = x0 + group.alpha.to_numpy() * w
            if quantity == "drift":
                observed = group.mean_observed_drift_per_update.to_numpy()
                predicted = group.interior_predicted_drift_per_update.to_numpy()
                ys = y0 + (observed - y_min) / (y_max - y_min) * h
                yp = y0 + (predicted - y_min) / (y_max - y_min) * h
                legacy.draw_polyline(c, xs, yp, color, 0.9, [3, 2])
                legacy.draw_polyline(c, xs, ys, color, 1.7)
            else:
                for field, dash in (("homophilic_acceptance_rate", None), ("reverse_acceptance_rate", [3, 2])):
                    valid = group[np.isfinite(group[field])]
                    if valid.empty:
                        continue
                    xx = x0 + valid.alpha.to_numpy() * w
                    yy = y0 + (valid[field].to_numpy() - y_min) / (y_max - y_min) * h
                    legacy.draw_polyline(c, xx, yy, color, 1.4, dash)
        c.setStrokeColor(black); c.rect(x0, y0, w, h, fill=0, stroke=1)
        font(c, 6.2); c.setFillColor(black)
        for tick in (0.0, 0.5, 1.0): c.drawCentredString(x0+tick*w, y0-11, f"{tick:.1f}")
        c.drawCentredString(x0+w/2, y0-23, "current mediation alpha")
        c.saveState(); c.translate(x0-26, y0+h/2); c.rotate(90); c.drawCentredString(0,0,title); c.restoreState()
        panel_label(c, chr(97+panel), x0-.31*inch, y0+h+.16*inch)
    font(c, 6.2); c.setFillColor(GREY)
    c.drawString(.72*inch, 2.71*inch, "solid = simulated mean; dashed = interior Poisson prediction")
    c.drawString(4.24*inch, 2.71*inch, "solid = homophilic; dashed = reverse")
    c.drawCentredString(width/2, .13*inch, "32 paired initializations; boundary rejections retained")
    c.showPage(); c.save()


def font(c: canvas.Canvas, size: float, bold: bool = False) -> None:
    legacy.set_font(c, size, bold)


def panel_label(c: canvas.Canvas, label: str, x: float, y: float) -> None:
    font(c, 10.0, True)
    c.setFillColor(black)
    c.drawString(x, y, label)


def quantile_bar(c: canvas.Canvas, x: float, values: np.ndarray, y0: float, h: float, ymax: float, color) -> None:
    q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
    # Lift exact zeros just above the frame so a closed-loop null remains visible.
    sy = lambda value: y0 + max(1.4, float(value) / ymax * h)
    c.setStrokeColor(color); c.setLineWidth(1.2); c.line(x, sy(q05), x, sy(q95))
    c.setFillColor(Color(color.red, color.green, color.blue, alpha=.20))
    c.rect(x-.09*inch, sy(q25), .18*inch, sy(q75)-sy(q25), fill=1, stroke=1)
    c.setStrokeColor(color); c.setLineWidth(1.8); c.line(x-.09*inch, sy(q50), x+.09*inch, sy(q50))


def figure(path: Path, gaps: pd.DataFrame, summaries: pd.DataFrame) -> None:
    width, height = 7.1 * inch, 5.20 * inch
    c = canvas.Canvas(str(path), pagesize=(width, height), initialFontName="AdaptiveSans")
    short = {
        "no rewiring": "null",
        "fast reversible": "fast",
        "fast reversible, slow ramp": "fast/slow",
        "slow reversible": "slow",
        "slow reversible, slow ramp": "slow/slow",
        "one-way boundary": "one-way",
    }
    ordered = list(short)
    # Panels a-b: distributions across seeds.
    for panel, metric, x0, color, ymax in (
        ("a", "Q", .62*inch, ORANGE, .42),
        ("b", "r_x", 3.93*inch, BLUE, .62),
    ):
        y0, w, h = 3.02*inch, 2.60*inch, 1.56*inch
        c.setFillColor(PALE); c.rect(x0, y0, w, h, fill=1, stroke=0)
        for i, regime in enumerate(ordered):
            xx = x0 + (i+.5)*w/len(ordered)
            vals = summaries[(summaries.regime==regime)&(summaries.metric==metric)].absolute_area.to_numpy()
            quantile_bar(c, xx, vals, y0, h, ymax, color)
            label_lines = {
                "null": ("null", ""), "fast": ("fast", ""),
                "fast/slow": ("fast", "slow ramp"), "slow": ("slow", ""),
                "slow/slow": ("slow", "slow ramp"), "one-way": ("one-way", ""),
            }[short[regime]]
            font(c,6.2); c.setFillColor(black)
            c.drawCentredString(xx,y0-10,label_lines[0])
            if label_lines[1]: c.drawCentredString(xx,y0-18,label_lines[1])
        c.setStrokeColor(black); c.rect(x0,y0,w,h,fill=0,stroke=1)
        font(c,6.5); c.saveState(); c.translate(x0-24,y0+h/2); c.rotate(90); c.drawCentredString(0,0,f"absolute loop area: {metric}"); c.restoreState()
        panel_label(c,panel,x0-.30*inch,y0+h+.12*inch)
        font(c,6.2); c.setFillColor(GREY); c.drawCentredString(x0+w/2,y0+h+5,"5--95% whisker; interquartile box; median")

    # Panel c: median branch-gap curves for the slowly reversible mechanism.
    x0,y0,w,h=.72*inch,.55*inch,2.75*inch,1.55*inch
    c.setFillColor(PALE); c.rect(x0,y0,w,h,fill=1,stroke=0)
    for metric,color,dash in (("Q",ORANGE,None),("r_x",BLUE,[4,2])):
        for regime,width_line in (("slow reversible",1.5),("slow reversible, slow ramp",2.4)):
            pivot = gaps[gaps.regime==regime].groupby("alpha")[f"delta_{metric}"].median().reset_index()
            xs=x0+pivot.alpha.to_numpy()*w
            vals=pivot[f"delta_{metric}"].to_numpy(); ymin,ymax=-.12,.70
            ys=y0+(vals-ymin)/(ymax-ymin)*h
            legacy.draw_polyline(c,xs,ys,color,width_line,dash)
    zero=y0+(0+.12)/.82*h; c.setStrokeColor(LIGHT); c.line(x0,zero,x0+w,zero)
    c.setStrokeColor(black); c.rect(x0,y0,w,h,fill=0,stroke=1)
    font(c,6.3); c.setFillColor(black); c.drawCentredString(x0+w/2,y0-12,"current mediation alpha")
    c.saveState(); c.translate(x0-21,y0+h/2); c.rotate(90); c.drawCentredString(0,0,"branch gap") ; c.restoreState()
    font(c,6.2); c.setFillColor(GREY); c.drawCentredString(x0+w/2,y0+h+5,"thin = dwell 30; thick = dwell 120")
    panel_label(c,"c",.38*inch,y0+h+.12*inch)

    # Panel d: graph restoration can close topology while node-state gaps remain.
    x0,y0,w,h=4.15*inch,.55*inch,2.20*inch,1.55*inch
    c.setFillColor(PALE); c.rect(x0,y0,w,h,fill=1,stroke=0)
    reset=summaries[summaries.regime=="slow reversible + graph reset"]
    colors={"Q":ORANGE,"r_x":BLUE,"phi":GREEN,"P":PURPLE}
    ymax=.35
    for i,metric in enumerate(METRICS):
        vals=np.abs(reset[reset.metric==metric].gap_at_alpha_0.to_numpy())
        xx=x0+(i+.5)*w/len(METRICS); quantile_bar(c,xx,vals,y0,h,ymax,colors[metric])
        font(c,6.5); c.setFillColor(black); c.drawCentredString(xx,y0-11,metric)
    c.setStrokeColor(black); c.rect(x0,y0,w,h,fill=0,stroke=1)
    font(c,6.3); c.setFillColor(black); c.saveState(); c.translate(x0-20,y0+h/2); c.rotate(90); c.drawCentredString(0,0,"endpoint gap"); c.restoreState()
    font(c,6.2); c.setFillColor(GREY); c.drawCentredString(x0+w/2,y0+h+5,"absolute endpoint gap after exact graph reset")
    panel_label(c,"d",3.80*inch,y0+h+.12*inch)

    font(c,6.2); c.setFillColor(GREY)
    c.drawRightString(width-.18*inch,.12*inch,"32 independent graph/opinion initializations per regime")
    c.showPage(); c.save()


def main() -> None:
    for directory in (DATA, FIGURES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    trajectories=[]; gap_frames=[]; summary_frames=[]
    for seed in range(20270824,20270856):
        for regime in REGIMES:
            frame=simulate(seed,regime); trajectories.append(frame)
            gaps,summaries=summarize(frame); gap_frames.append(gaps); summary_frames.append(summaries)
    trajectory_frame=pd.concat(trajectories,ignore_index=True)
    gap_frame=pd.concat(gap_frames,ignore_index=True)
    summary_frame=pd.concat(summary_frames,ignore_index=True)
    validation_frame=drift_validation(trajectory_frame)
    trajectory_frame.to_csv(DATA/"adaptive_network_stress_trajectories.csv",index=False)
    gap_frame.to_csv(DATA/"adaptive_network_stress_gaps.csv",index=False)
    summary_frame.to_csv(DATA/"adaptive_network_stress_metrics.csv",index=False)
    validation_frame.to_csv(DATA/"adaptive_network_bridge_drift_validation.csv",index=False)
    figure(FIGURES/"fig6_adaptive_network.pdf",gap_frame,summary_frame)
    drift_figure(FIGURES/"fig8_bridge_validation.pdf",validation_frame)
    summary={
        "status":"synthetic reversible-network stress test; not an empirical estimate",
        "base_network_parameters":asdict(legacy.BASE),
        "n_initializations":32,
        "seed_range":[20270824,20270855],
        "regimes":[asdict(regime) for regime in REGIMES],
        "metric_definitions":{
            "Q":"Newman-Girvan modularity of the fixed two-camp partition.",
            "r_x":"Pearson correlation of node opinions across oriented edge endpoints.",
            "phi":"Cross-camp conductance; not independent of Q in the planted-cut design.",
            "P":"Mean absolute node opinion.",
        },
        "analytical_bridge_count_result":{
            "exact_conditional_drift":"2 * (E[accepted reverse swaps | graph, alpha] - E[accepted homophilic swaps | graph, alpha])",
            "interior_drift":"2*N*(rho_minus*(alpha0-alpha)_+/alpha0 - rho_plus*(alpha-alpha0)_+/(1-alpha0))",
            "ideal_finite_horizon_repair_probability":"Pr[Poisson(lambda_minus*T) >= ceil(deficit/2)]",
            "actual_probability_bound":"actual repair probability is no larger than the ideal Poisson tail when proposals can fail",
            "ideal_mean_repair_time":"ceil(deficit/2)/lambda_minus; a lower bound with feasibility failures",
            "boundary_note":"Proposal failures and the simple-graph constraint reduce accepted events; rho_minus=0 makes the cross-edge deficit absorbing under this topology rule.",
        },
        "opinion_noise":{
            "law":"iid Normal(0, 0.0035^2) added inside the [-1,1] projection after each deterministic update",
            "independence":"independent over nodes and updates",
            "stream":"dedicated common-random-number stream within each initialization; rewiring uses a separate stream",
        },
        "bridge_drift_validation":validation_frame.to_dict(orient="records"),
        "quantiles":quantile_summary(summary_frame),
        "invariants":[
            "All adjacency matrices are simple, undirected, and loop free.",
            "Every stochastic swap preserves each node degree exactly.",
            "Each condition within a seed starts from the same graph and opinion state.",
            "The Hamiltonian-cycle initializer guarantees degree at least two and rewiring preserves that degree sequence.",
            "Reverse rewiring restores mixing toward the initial cross-edge count without restoring edge identities.",
        ],
    }
    (RESULTS/"adaptive_network_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"status":summary["status"],"runs":len(trajectories),"trajectory_rows":len(trajectory_frame)},indent=2))


if __name__=="__main__":
    main()
