"""Reproduce all simulations and vector figures for the manuscript.

The model is a deliberately reduced-order theory. It does not estimate real-world
effects. Its purpose is to expose falsifiable implications of distinct memory
channels under a closed-loop intervention protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import reportlab
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

BLUE = HexColor("#1769AA")
ORANGE = HexColor("#E07A1F")
GREEN = HexColor("#2A8C69")
PURPLE = HexColor("#7C4D9E")
RED = HexColor("#B33A3A")
GREY = HexColor("#666B73")
LIGHT_GREY = HexColor("#D9DEE5")
PALE = HexColor("#F4F6F8")

_REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("FigureSans", _REPORTLAB_FONTS / "Vera.ttf"))
pdfmetrics.registerFont(TTFont("FigureSans-Bold", _REPORTLAB_FONTS / "VeraBd.ttf"))


@dataclass(frozen=True)
class Parameters:
    gain: float = 10.0
    w_g: float = 0.45
    w_alpha: float = 0.60
    w_m: float = 0.40
    w_l: float = 0.18
    w_c: float = 0.12
    threshold: float = 0.62
    tau_g: float = 1.0
    tau_m: float = 30.0
    tau_l: float = 40.0
    tau_c: float = 60.0
    delta_m: float = 0.03
    build_0: float = 0.04
    build_alpha: float = 0.18
    decay_l: float = 0.04
    consolidate_c: float = 0.12
    recover_c: float = 0.045


BASE = Parameters()

# The robustness experiment intentionally varies 14 of the 17 parameters.
# tau_g fixes the time unit, while build_0 and build_alpha are held fixed so
# that the local perturbation probes response and memory strength without also
# changing the baseline topology-formation experiment.
ROBUSTNESS_PARAMETERS = (
    "gain",
    "w_g",
    "w_alpha",
    "w_m",
    "w_l",
    "w_c",
    "threshold",
    "tau_m",
    "tau_l",
    "tau_c",
    "delta_m",
    "decay_l",
    "consolidate_c",
    "recover_c",
)
ROBUSTNESS_FIXED_PARAMETERS = ("tau_g", "build_0", "build_alpha")


def logistic(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def advance(
    state: np.ndarray,
    alpha: float,
    p: Parameters,
    dt: float,
    neutralized: frozenset[str] = frozenset(),
) -> np.ndarray:
    """Advance one Euler step; dt=0.1 is stable for the selected time scales."""
    g, m, ell, cog = state
    m_eff = 0.0 if "M" in neutralized else m
    l_eff = 0.0 if "L" in neutralized else ell
    c_eff = 0.0 if "C" in neutralized else cog
    field = (
        p.w_g * g
        + p.w_alpha * alpha
        + p.w_m * alpha * m_eff
        + p.w_l * l_eff
        + p.w_c * c_eff
        - p.threshold
    )
    target = logistic(p.gain * field)
    dg = (target - g) / p.tau_g
    dm = (alpha * (g - m) - p.delta_m * (1.0 - alpha) * m) / p.tau_m
    dl = (
        (p.build_0 + p.build_alpha * alpha) * g * (1.0 - ell)
        - p.decay_l * (1.0 - g) * ell
    ) / p.tau_l
    dc = (
        p.consolidate_c * g * (1.0 - cog)
        - p.recover_c * (1.0 - g) * cog
    ) / p.tau_c
    updated = state + dt * np.array([dg, dm, dl, dc], dtype=float)
    return np.clip(updated, 0.0, 1.0)


def simulate_protocol(
    p: Parameters = BASE,
    dwell: float = 50.0,
    n_levels: int = 41,
    dt: float = 0.1,
    neutralized: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    up = np.linspace(0.0, 1.0, n_levels)
    down = np.linspace(1.0, 0.0, n_levels)[1:]
    schedule = [("up", float(a)) for a in up] + [("down", float(a)) for a in down]
    state = np.array([0.001, 0.0, 0.0, 0.0], dtype=float)
    rows = []
    n_steps = max(1, int(round(dwell / dt)))
    for step, (branch, alpha) in enumerate(schedule):
        for _ in range(n_steps):
            state = advance(state, alpha, p, dt, neutralized)
        rows.append(
            {
                "step": step,
                "branch": branch,
                "alpha": alpha,
                "G": state[0],
                "M": state[1],
                "L": state[2],
                "C": state[3],
                "dwell": dwell,
                "neutralized": "+".join(sorted(neutralized)) or "none",
            }
        )
    return pd.DataFrame(rows)


def aligned_branches(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    up = frame[frame.branch == "up"].sort_values("alpha").reset_index(drop=True)
    peak = up.iloc[[-1]].copy()
    peak.loc[:, "branch"] = "down"
    down = pd.concat([frame[frame.branch == "down"], peak], ignore_index=True)
    down = down.sort_values("alpha").reset_index(drop=True)
    return up, down


def hysteresis_metrics(frame: pd.DataFrame) -> dict[str, float]:
    up, down = aligned_branches(frame)
    gap = down.G.to_numpy() - up.G.to_numpy()
    alpha = up.alpha.to_numpy()
    area = float(np.trapezoid(gap, alpha))
    max_gap_index = int(np.argmax(np.abs(gap)))

    def threshold_cross(branch: pd.DataFrame, direction: str) -> float:
        ordered = branch.sort_values("alpha", ascending=(direction == "up"))
        if direction == "up":
            matches = ordered[ordered.G >= 0.5]
        else:
            matches = ordered[ordered.G < 0.5]
        return float(matches.iloc[0].alpha) if not matches.empty else float("nan")

    alpha_up = threshold_cross(up, "up")
    alpha_down = threshold_cross(down, "down")
    return {
        "area": area,
        "max_gap": float(gap[max_gap_index]),
        "alpha_at_max_gap": float(alpha[max_gap_index]),
        "alpha_up_half": alpha_up,
        "alpha_down_half": alpha_down,
        "threshold_width": alpha_up - alpha_down,
    }


def rate_sweep() -> pd.DataFrame:
    rows = []
    for dwell in (2, 5, 10, 25, 50, 100, 250, 500):
        for neutralized in (frozenset(), frozenset({"M", "L", "C"})):
            frame = simulate_protocol(dwell=float(dwell), neutralized=neutralized)
            rows.append(
                {
                    "dwell": dwell,
                    "condition": "baseline" if not neutralized else "all channels neutralized",
                    **hysteresis_metrics(frame),
                }
            )
    return pd.DataFrame(rows)


def channel_ablation() -> pd.DataFrame:
    rows = []
    channels = ("M", "L", "C")
    for n in range(4):
        for subset in itertools.combinations(channels, n):
            neutralized = frozenset(subset)
            frame = simulate_protocol(dwell=50.0, neutralized=neutralized)
            rows.append(
                {
                    "neutralized": "+".join(subset) or "none",
                    "n_channels": n,
                    **hysteresis_metrics(frame),
                }
            )
    return pd.DataFrame(rows).sort_values(["n_channels", "neutralized"]).reset_index(drop=True)


def sensitivity_grid() -> pd.DataFrame:
    rows = []
    gains = np.linspace(7.0, 13.0, 17)
    feedbacks = np.linspace(0.30, 0.62, 17)
    for gain in gains:
        for w_g in feedbacks:
            p = replace(BASE, gain=float(gain), w_g=float(w_g))
            frame = simulate_protocol(p=p, dwell=25.0, n_levels=31, dt=0.2)
            rows.append({"gain": gain, "w_g": w_g, **hysteresis_metrics(frame)})
    return pd.DataFrame(rows)


def robustness_sample(n: int = 400, seed: int = 20270824) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    varied = list(ROBUSTNESS_PARAMETERS)
    base_map = asdict(BASE)
    for draw in range(n):
        values = dict(base_map)
        multipliers = rng.uniform(0.8, 1.2, len(varied))
        for key, multiplier in zip(varied, multipliers):
            values[key] = base_map[key] * float(multiplier)
        p = Parameters(**values)
        full = hysteresis_metrics(
            simulate_protocol(p=p, dwell=25.0, n_levels=31, dt=0.2)
        )["area"]
        neutral = hysteresis_metrics(
            simulate_protocol(
                p=p,
                dwell=25.0,
                n_levels=31,
                dt=0.2,
                neutralized=frozenset({"M", "L", "C"}),
            )
        )["area"]
        rows.append(
            {
                "draw": draw,
                "area_baseline": full,
                "area_neutralized": neutral,
                "memory_attributable_area": full - neutral,
                **{f"multiplier_{key}": value for key, value in zip(varied, multipliers)},
            }
        )
    return pd.DataFrame(rows)


def ablation_lattice(ablation: pd.DataFrame, tolerance: float = 0.05) -> dict:
    """Exact set-function decomposition of whole-path mechanism ablations."""
    channels = ("M", "L", "C")

    def bundle(label: str) -> frozenset[str]:
        return frozenset() if label == "none" else frozenset(label.split("+"))

    deficit = {
        bundle(row.neutralized): float(row.area)
        for row in ablation.itertuples(index=False)
    }
    empty = frozenset()
    benefit = {subset: deficit[empty] - value for subset, value in deficit.items()}
    interactions = {}
    for size in range(1, len(channels) + 1):
        for subset_tuple in itertools.combinations(channels, size):
            subset = frozenset(subset_tuple)
            eta = 0.0
            for inner_size in range(size + 1):
                for inner_tuple in itertools.combinations(subset_tuple, inner_size):
                    inner = frozenset(inner_tuple)
                    eta += (-1.0) ** (size - inner_size) * benefit[inner]
            interactions["+".join(subset_tuple)] = eta
    feasible = [subset for subset, value in deficit.items() if value <= tolerance]
    minimum_size = min((len(subset) for subset in feasible), default=None)
    minimum_bundles = sorted(
        "+".join(sorted(subset)) or "none"
        for subset in feasible
        if len(subset) == minimum_size
    )
    return {
        "deficit": {
            "+".join(sorted(subset)) or "none": value
            for subset, value in sorted(deficit.items(), key=lambda item: (len(item[0]), sorted(item[0])))
        },
        "benefit": {
            "+".join(sorted(subset)) or "none": value
            for subset, value in sorted(benefit.items(), key=lambda item: (len(item[0]), sorted(item[0])))
        },
        "mobius_interactions": interactions,
        "tolerance": tolerance,
        "minimum_ablation_sets_under_path_tolerance": minimum_bundles,
        "minimum_ablation_set_size": minimum_size,
        "scope": "Whole-path mechanism ablations, not time-indexed restoration policies; this object does not solve reversal controllability.",
    }


def set_font(c: canvas.Canvas, size: float, bold: bool = False) -> None:
    c.setFont("FigureSans-Bold" if bold else "FigureSans", size)


def title(c: canvas.Canvas, text: str, width: float, height: float) -> None:
    set_font(c, 11.2, True)
    c.setFillColor(black)
    c.drawString(0.42 * inch, height - 0.38 * inch, text)


def panel_label(c: canvas.Canvas, label: str, x: float, y: float) -> None:
    set_font(c, 10.5, True)
    c.setFillColor(black)
    c.drawString(x, y, label)


def draw_axes(
    c: canvas.Canvas,
    rect: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    xticks: list[float],
    yticks: list[float],
    xformatter=lambda value: f"{value:g}",
    yformatter=lambda value: f"{value:g}",
) -> tuple:
    x, y, w, h = rect
    c.setStrokeColor(LIGHT_GREY)
    c.setLineWidth(0.45)
    for tick in yticks:
        py = y + (tick - ylim[0]) / (ylim[1] - ylim[0]) * h
        c.line(x, py, x + w, py)
    c.setStrokeColor(black)
    c.setLineWidth(0.75)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    set_font(c, 7.3)
    for tick in xticks:
        px = x + (tick - xlim[0]) / (xlim[1] - xlim[0]) * w
        c.line(px, y, px, y - 3)
        label = xformatter(tick)
        c.drawCentredString(px, y - 12, label)
    for tick in yticks:
        py = y + (tick - ylim[0]) / (ylim[1] - ylim[0]) * h
        c.line(x - 3, py, x, py)
        c.drawRightString(x - 5, py - 2.5, yformatter(tick))
    set_font(c, 8.2)
    c.drawCentredString(x + w / 2, y - 25, xlabel)
    c.saveState()
    c.translate(x - 32, y + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()

    def transform(xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        px = x + (xs - xlim[0]) / (xlim[1] - xlim[0]) * w
        py = y + (ys - ylim[0]) / (ylim[1] - ylim[0]) * h
        return px, py

    return transform


def draw_polyline(c: canvas.Canvas, xs, ys, color, width=1.6, dash=None) -> None:
    path = c.beginPath()
    for index, (x, y) in enumerate(zip(xs, ys)):
        if index == 0:
            path.moveTo(float(x), float(y))
        else:
            path.lineTo(float(x), float(y))
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.drawPath(path)
    c.setDash([])


def draw_legend(c: canvas.Canvas, items, x: float, y: float, spacing: float = 15.0) -> None:
    set_font(c, 7.4)
    for index, (label, color, dash) in enumerate(items):
        yy = y - index * spacing
        c.setStrokeColor(color)
        c.setLineWidth(1.8)
        c.setDash(dash or [])
        c.line(x, yy, x + 18, yy)
        c.setDash([])
        c.setFillColor(black)
        c.drawString(x + 23, yy - 2.5, label)


def fig1_memory_ledger(path: Path) -> None:
    width, height = 7.1 * inch, 4.35 * inch
    c = canvas.Canvas(
        str(path), pagesize=(width, height), initialFontName="FigureSans"
    )
    # Captions are supplied by the manuscript; artwork contains no title.
    set_font(c, 8.2)
    c.setFillColor(GREY)
    c.drawString(0.44 * inch, height - 0.63 * inch, "Current treatment is not current state.")

    boxes = [
        (0.50, 2.16, 1.35, 0.76, "Recommender", "profile M", BLUE),
        (2.10, 2.16, 1.35, 0.76, "Follow graph", "topology L", GREEN),
        (3.70, 2.16, 1.35, 0.76, "Human/norm", "memory C", PURPLE),
        (5.35, 2.16, 1.25, 0.76, "Collective", "state G", ORANGE),
    ]
    for x_i, y_i, w_i, h_i, line1, line2, color in boxes:
        x, y, w, h = x_i * inch, y_i * inch, w_i * inch, h_i * inch
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.11))
        c.setStrokeColor(color)
        c.setLineWidth(1.2)
        c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
        c.setFillColor(black)
        set_font(c, 8.4, True)
        c.drawCentredString(x + w / 2, y + h * 0.60, line1)
        set_font(c, 8.2)
        c.drawCentredString(x + w / 2, y + h * 0.30, line2)

    def arrow(x1, y1, x2, y2, color=GREY):
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.1)
        c.line(x1 * inch, y1 * inch, x2 * inch, y2 * inch)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 0.08 * inch
        for offset in (0.45, -0.45):
            c.line(
                x2 * inch,
                y2 * inch,
                x2 * inch - size * math.cos(angle + offset),
                y2 * inch - size * math.sin(angle + offset),
            )

    arrow(1.85, 2.54, 2.10, 2.54)
    arrow(3.45, 2.54, 3.70, 2.54)
    arrow(5.05, 2.54, 5.35, 2.54)
    arrow(5.90, 2.13, 1.18, 1.57, HexColor("#9AA1AA"))
    arrow(5.98, 2.13, 2.75, 1.56, HexColor("#9AA1AA"))
    arrow(6.08, 2.13, 4.36, 1.56, HexColor("#9AA1AA"))

    c.setFillColor(PALE)
    c.setStrokeColor(LIGHT_GREY)
    c.roundRect(0.48 * inch, 0.56 * inch, 6.14 * inch, 0.73 * inch, 6, fill=1, stroke=1)
    set_font(c, 8.2, True)
    c.setFillColor(black)
    c.drawString(0.67 * inch, 1.04 * inch, "Closed-loop estimand")
    set_font(c, 8.1)
    c.drawString(0.67 * inch, 0.76 * inch, "Delta G(alpha) = G_down(alpha) - G_up(alpha), matched on current alpha")
    c.setFillColor(RED)
    set_font(c, 7.3, True)
    c.drawRightString(6.42 * inch, 1.04 * inch, "Switch-off does not restore state")
    c.showPage()
    c.save()


def fig2_closed_loop(path: Path, baseline: pd.DataFrame) -> None:
    width, height = 7.1 * inch, 4.45 * inch
    c = canvas.Canvas(
        str(path), pagesize=(width, height), initialFontName="FigureSans"
    )
    # Captions are supplied by the manuscript; artwork contains no title.
    up, down = aligned_branches(baseline)

    left = (0.72 * inch, 0.88 * inch, 2.45 * inch, 2.79 * inch)
    tr = draw_axes(c, left, (0, 1), (0, 1.02), "mediation level, alpha", "collective state, G", [0, .25, .5, .75, 1], [0, .25, .5, .75, 1])
    xu, yu = tr(up.alpha, up.G)
    xd, yd = tr(down.alpha, down.G)
    area = c.beginPath()
    area.moveTo(xu[0], yu[0])
    for x, y in zip(xu[1:], yu[1:]):
        area.lineTo(float(x), float(y))
    for x, y in zip(xd[::-1], yd[::-1]):
        area.lineTo(float(x), float(y))
    area.close()
    c.setFillColor(Color(ORANGE.red, ORANGE.green, ORANGE.blue, alpha=0.13))
    c.setStrokeColor(Color(1, 1, 1, alpha=0))
    c.drawPath(area, fill=1, stroke=0)
    draw_polyline(c, xu, yu, BLUE, 1.9)
    draw_polyline(c, xd, yd, ORANGE, 1.9)
    draw_legend(c, [("up branch", BLUE, None), ("down branch", ORANGE, None)], 0.93 * inch, 3.42 * inch)
    panel_label(c, "a", 0.40 * inch, 3.80 * inch)
    metrics = hysteresis_metrics(baseline)
    set_font(c, 7.3)
    c.setFillColor(GREY)
    c.drawString(1.10 * inch, 0.16 * inch, f"loop area H = {metrics['area']:.3f}")

    right = (4.03 * inch, 0.88 * inch, 2.35 * inch, 2.79 * inch)
    tr2 = draw_axes(c, right, (0, 1), (-0.02, 1.02), "mediation level, alpha", "descending minus ascending", [0, .25, .5, .75, 1], [0, .25, .5, .75, 1])
    for variable, color in (("M", BLUE), ("L", GREEN), ("C", PURPLE)):
        gap = down[variable].to_numpy() - up[variable].to_numpy()
        x, y = tr2(up.alpha, gap)
        draw_polyline(c, x, y, color, 1.7)
    draw_legend(c, [("profile M", BLUE, None), ("topology L", GREEN, None), ("human/norm C", PURPLE, None)], 4.31 * inch, 3.43 * inch)
    panel_label(c, "b", 3.70 * inch, 3.80 * inch)
    set_font(c, 7.3)
    c.setFillColor(GREY)
    c.drawCentredString(5.18 * inch, 0.16 * inch, "same alpha, different internal state")
    c.showPage()
    c.save()


def fig3_rate_ablation(path: Path, rates: pd.DataFrame, ablation: pd.DataFrame) -> None:
    width, height = 7.1 * inch, 4.45 * inch
    c = canvas.Canvas(
        str(path), pagesize=(width, height), initialFontName="FigureSans"
    )
    # Captions are supplied by the manuscript; artwork contains no title.

    left = (0.74 * inch, 0.88 * inch, 2.52 * inch, 2.79 * inch)
    tr = draw_axes(c, left, (math.log10(2), math.log10(500)), (0, 0.52), "dwell per alpha level (log scale)", "absolute path deficit", [math.log10(v) for v in (2, 5, 10, 25, 50, 100, 250, 500)], [0, .1, .2, .3, .4, .5], xformatter=lambda v: f"{int(round(10**v))}")
    for condition, color, dash in (("baseline", ORANGE, None), ("all channels neutralized", GREY, [4, 3])):
        group = rates[rates.condition == condition].sort_values("dwell")
        x, y = tr(np.log10(group.dwell), group.area)
        draw_polyline(c, x, y, color, 1.9, dash)
        c.setFillColor(color)
        for xx, yy in zip(x, y):
            c.circle(float(xx), float(yy), 2.2, fill=1, stroke=0)
    draw_legend(c, [("baseline", ORANGE, None), ("M+L+C ablated", GREY, [4, 3])], 0.96 * inch, 3.44 * inch)
    panel_label(c, "a", 0.40 * inch, 3.80 * inch)

    right_x, right_y, right_w, right_h = 3.88 * inch, 0.88 * inch, 2.58 * inch, 2.79 * inch
    panel_label(c, "b", 3.55 * inch, 3.80 * inch)
    ordered = ablation.sort_values("area", ascending=True).reset_index(drop=True)
    labels = [value.replace("none", "no ablation") for value in ordered.neutralized]
    c.setFillColor(PALE)
    c.rect(right_x, right_y, right_w, right_h, fill=1, stroke=0)
    for tick in (0, .1, .2, .3, .4, .5):
        xx = right_x + tick / .5 * right_w
        c.setStrokeColor(LIGHT_GREY)
        c.line(xx, right_y, xx, right_y + right_h)
        set_font(c, 7.1)
        c.setFillColor(black)
        c.drawCentredString(xx, right_y - 12, f"{tick:.1f}")
    bar_h = right_h / (len(ordered) + 1.2)
    for index, row in ordered.iterrows():
        yy = right_y + (index + 0.65) * bar_h
        color = RED if row.neutralized == "M+L+C" else (ORANGE if row.neutralized == "none" else BLUE)
        c.setFillColor(color)
        c.rect(right_x, yy, float(row.area) / .5 * right_w, bar_h * .58, fill=1, stroke=0)
        set_font(c, 6.9)
        c.setFillColor(black)
        c.drawRightString(right_x - 5, yy + 1, labels[index])
        c.drawString(right_x + float(row.area) / .5 * right_w + 4, yy + 1, f"{row.area:.3f}")
    set_font(c, 8.2)
    c.drawCentredString(right_x + right_w / 2, right_y - 25, "absolute path deficit")
    set_font(c, 7.3)
    c.setFillColor(GREY)
    c.drawCentredString(5.16 * inch, 0.16 * inch, "whole-path mechanism ablations")
    c.showPage()
    c.save()


def viridis_like(value: float) -> Color:
    value = max(0.0, min(1.0, value))
    anchors = [
        (0.0, HexColor("#F7FBFF")),
        (0.30, HexColor("#C6DBEF")),
        (0.55, HexColor("#6BAED6")),
        (0.78, HexColor("#2171B5")),
        (1.0, HexColor("#08306B")),
    ]
    for (x0, c0), (x1, c1) in zip(anchors[:-1], anchors[1:]):
        if x0 <= value <= x1:
            f = (value - x0) / (x1 - x0)
            return Color(
                c0.red + f * (c1.red - c0.red),
                c0.green + f * (c1.green - c0.green),
                c0.blue + f * (c1.blue - c0.blue),
            )
    return anchors[-1][1]


def fig4_sensitivity(path: Path, grid: pd.DataFrame, robust: pd.DataFrame) -> None:
    width, height = 7.1 * inch, 4.45 * inch
    c = canvas.Canvas(
        str(path), pagesize=(width, height), initialFontName="FigureSans"
    )
    # Captions are supplied by the manuscript; artwork contains no title.
    panel_label(c, "a", 0.40 * inch, 3.80 * inch)
    x0, y0, w, h = 0.84 * inch, 0.90 * inch, 2.45 * inch, 2.74 * inch
    gains = sorted(grid.gain.unique())
    wgs = sorted(grid.w_g.unique())
    cell_w, cell_h = w / len(gains), h / len(wgs)
    max_h = max(0.5, float(grid.area.max()))
    for ix, gain in enumerate(gains):
        for iy, wg in enumerate(wgs):
            value = float(grid[(grid.gain == gain) & (grid.w_g == wg)].iloc[0].area)
            c.setFillColor(viridis_like(value / max_h))
            c.rect(x0 + ix * cell_w, y0 + iy * cell_h, cell_w + .2, cell_h + .2, fill=1, stroke=0)
    c.setStrokeColor(black)
    c.setLineWidth(.75)
    c.rect(x0, y0, w, h, fill=0, stroke=1)
    set_font(c, 7.2)
    for gain in (7, 9, 11, 13):
        xx = x0 + (gain - min(gains)) / (max(gains) - min(gains)) * w
        c.drawCentredString(xx, y0 - 12, str(gain))
    for wg in (.30, .38, .46, .54, .62):
        yy = y0 + (wg - min(wgs)) / (max(wgs) - min(wgs)) * h
        c.drawRightString(x0 - 5, yy - 2.5, f"{wg:.2f}")
    set_font(c, 8.2)
    c.drawCentredString(x0 + w / 2, y0 - 25, "social response gain k")
    c.saveState()
    c.translate(x0 - 36, y0 + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "endogenous feedback w_G")
    c.restoreState()
    # Analytic bistability boundary k*w_G=4.
    xs, ys = [], []
    for gain in np.linspace(7, 13, 120):
        wg = 4.0 / gain
        if min(wgs) <= wg <= max(wgs):
            xs.append(x0 + (gain - min(gains)) / (max(gains) - min(gains)) * w)
            ys.append(y0 + (wg - min(wgs)) / (max(wgs) - min(wgs)) * h)
    draw_polyline(c, xs, ys, RED, 1.5, [4, 2])
    set_font(c, 7.1, True)
    c.setFillColor(RED)
    c.drawString(x0 + 0.08 * inch, y0 + h - 0.17 * inch, "k w_G = 4")
    # Compact horizontal color scale for hysteresis area.
    cb_x, cb_y, cb_w, cb_h = 1.55 * inch, 0.20 * inch, 1.22 * inch, 0.08 * inch
    for index in range(80):
        value = index / 79
        c.setFillColor(viridis_like(value))
        c.rect(cb_x + index / 80 * cb_w, cb_y, cb_w / 80 + .2, cb_h, fill=1, stroke=0)
    c.setStrokeColor(black)
    c.setLineWidth(.5)
    c.rect(cb_x, cb_y, cb_w, cb_h, fill=0, stroke=1)
    set_font(c, 6.6)
    c.setFillColor(black)
    c.drawRightString(cb_x - 5, cb_y, "H")
    c.drawCentredString(cb_x, cb_y - 10, "0")
    c.drawCentredString(cb_x + cb_w / 2, cb_y - 10, f"{max_h/2:.2f}")
    c.drawCentredString(cb_x + cb_w, cb_y - 10, f"{max_h:.2f}")

    panel_label(c, "b", 3.61 * inch, 3.80 * inch)
    rect = (4.01 * inch, 0.90 * inch, 2.34 * inch, 2.74 * inch)
    max_x = max(0.62, float(robust.area_baseline.max()))
    tr = draw_axes(c, rect, (0, max_x), (0, 1.0), "hysteresis area H", "empirical CDF", [0, .1, .2, .3, .4, .5, .6], [0, .25, .5, .75, 1])
    for column, color, dash, label in (
        ("area_baseline", ORANGE, None, "baseline"),
        ("area_neutralized", GREY, [4, 3], "M+L+C neutralized"),
    ):
        values = np.sort(robust[column].to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)
        x, y = tr(values, cdf)
        draw_polyline(c, x, y, color, 1.8, dash)
    draw_legend(c, [("baseline", ORANGE, None), ("M+L+C neutralized", GREY, [4, 3])], 4.24 * inch, 3.42 * inch)
    set_font(c, 7.2)
    c.setFillColor(GREY)
    q = robust.area_baseline.quantile([.05, .5, .95]).to_numpy()
    c.drawCentredString(5.18 * inch, 0.16 * inch, f"400 draws, +/-20%; baseline H 5/50/95% = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    c.showPage()
    c.save()


def fig5_experimental_design(path: Path) -> None:
    width, height = 7.1 * inch, 4.3 * inch
    c = canvas.Canvas(
        str(path), pagesize=(width, height), initialFontName="FigureSans"
    )
    # Captions are supplied by the manuscript; artwork contains no title.
    set_font(c, 7.8)
    c.setFillColor(GREY)
    c.drawString(0.45 * inch, height - 0.64 * inch, "Clusters are randomized to sequence and rate; alpha is matched across histories.")
    x0, x1 = 1.18 * inch, 6.45 * inch
    y_positions = [3.08, 2.39, 1.70, 1.01]
    labels = ["Up-down", "Down-up", "Slow ramp", "Reset bundle"]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    schedules = [
        [0, .2, .4, .6, .8, 1, .8, .6, .4, .2, 0],
        [1, .8, .6, .4, .2, 0, .2, .4, .6, .8, 1],
        [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0],
        [0, .2, .4, .6, .8, 1, .8, .6, .4, .2, 0],
    ]
    for index, (y_in, label, color, schedule) in enumerate(zip(y_positions, labels, colors, schedules)):
        y = y_in * inch
        set_font(c, 8.0, True)
        c.setFillColor(black)
        c.drawRightString(1.03 * inch, y - 2, label)
        c.setStrokeColor(LIGHT_GREY)
        c.line(x0, y, x1, y)
        xs = np.linspace(x0, x1, len(schedule))
        ys = y - .18 * inch + np.asarray(schedule) * .36 * inch
        draw_polyline(c, xs, ys, color, 1.8)
        c.setFillColor(color)
        for xx, yy in zip(xs, ys):
            c.circle(float(xx), float(yy), 2.1, fill=1, stroke=0)
        if index == 3:
            reset_x = xs[5]
            c.setStrokeColor(RED)
            c.setLineWidth(1.1)
            c.line(reset_x, y - .28 * inch, reset_x, y + .30 * inch)
            set_font(c, 6.8, True)
            c.setFillColor(RED)
            c.drawCentredString(reset_x, y + .34 * inch, "neutralize M/L")
    set_font(c, 7.4)
    c.setFillColor(GREY)
    c.drawCentredString((x0 + x1) / 2, 0.51 * inch, "period / stable exposure window")
    c.setFillColor(PALE)
    c.setStrokeColor(LIGHT_GREY)
    c.roundRect(0.46 * inch, 3.48 * inch, 6.12 * inch, .34 * inch, 5, fill=1, stroke=1)
    set_font(c, 7.5, True)
    c.setFillColor(black)
    c.drawCentredString(3.52 * inch, 3.59 * inch, "Primary estimand: branch x alpha; diagnostics: branch x rate and branch x reset")
    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 80 instead of 400 robustness draws")
    args = parser.parse_args()
    for directory in (DATA, FIGURES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)

    baseline = simulate_protocol()
    rates = rate_sweep()
    ablation = channel_ablation()
    grid = sensitivity_grid()
    robust = robustness_sample(n=80 if args.quick else 400)

    baseline.to_csv(DATA / "baseline_loop.csv", index=False)
    rates.to_csv(DATA / "rate_sweep.csv", index=False)
    ablation.to_csv(DATA / "channel_neutralization.csv", index=False)
    grid.to_csv(DATA / "sensitivity_grid.csv", index=False)
    robust.to_csv(DATA / "robustness_sample.csv", index=False)

    baseline_metrics = hysteresis_metrics(baseline)
    neutral_metrics = hysteresis_metrics(
        simulate_protocol(neutralized=frozenset({"M", "L", "C"}))
    )
    quantiles = robust.area_baseline.quantile([.05, .5, .95]).to_dict()
    summary = {
        "model": asdict(BASE),
        "protocol": {"levels": 41, "dwell": 50.0, "dt": 0.1, "alpha_max": 1.0},
        "baseline": baseline_metrics,
        "all_channels_ablated": neutral_metrics,
        "slowest_ramp_area": float(
            rates[(rates.dwell == 500) & (rates.condition == "baseline")].iloc[0].area
        ),
        "robustness": {
            "n": len(robust),
            "varied_parameters": list(ROBUSTNESS_PARAMETERS),
            "fixed_parameters": list(ROBUSTNESS_FIXED_PARAMETERS),
            "fraction_baseline_area_gt_0_05": float((robust.area_baseline > .05).mean()),
            "fraction_memory_attributable_gt_0_05": float((robust.memory_attributable_area > .05).mean()),
            "baseline_area_quantiles": {str(key): float(value) for key, value in quantiles.items()},
        },
        "ablation_lattice": ablation_lattice(ablation),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig1_memory_ledger(FIGURES / "fig1_memory_ledger.pdf")
    fig2_closed_loop(FIGURES / "fig2_closed_loop.pdf", baseline)
    fig3_rate_ablation(FIGURES / "fig3_rate_ablation.pdf", rates, ablation)
    fig4_sensitivity(FIGURES / "fig4_sensitivity.pdf", grid, robust)
    fig5_experimental_design(FIGURES / "fig5_experimental_design.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
