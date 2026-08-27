"""Fail-fast scientific and packaging checks for the release archive."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived"
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    report: dict[str, object] = {"status": "passed", "checks": {}}
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]

    manuscript = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    selected = pd.read_csv(ROOT / "literature" / "selected_sources.csv", dtype=str).fillna("")
    require(len(selected) == 107, "selected bibliography must contain 107 records")
    require(selected.key.is_unique and selected.doi.str.lower().is_unique, "duplicate key or DOI")
    cited: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript):
        for key in group.split(","):
            key = key.strip()
            if key and key not in cited:
                cited.append(key)
    require(set(cited) == set(selected.key), "manuscript citations and selected registry diverge")
    checks["bibliography"] = {"records": len(selected), "citation_closure": True}

    stale = [
        "Identifying Algorithmically Mediated Social Hysteresis",
        "Hysteresis relative to $\\mathfrak L$",
        "a claim relative to $\\mathfrak L$",
        "davidson2025closedloop",
        "10.48550/arxiv.2507.19792",
        "one prespecified indexed cluster",
        "4,000 independent null replicates",
        "dense numerical continuation grid",
    ]
    require(not any(term in manuscript for term in stale), "stale claim remains in manuscript")
    require("Testing Algorithmically Mediated Social Hysteresis" in manuscript, "release title missing")
    require("10.1109/TCSS.2026.3715162" in (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8"),
            "Davidson--Ye Version of Record missing")

    expected_figures = {
        "fig1_memory_ledger.pdf", "fig2_causal_dag.pdf", "fig2_closed_loop.pdf",
        "fig3_rate_ablation.pdf", "fig4_sensitivity.pdf", "fig5_bifurcation.pdf",
        "fig5_experimental_design.pdf", "fig6_adaptive_network.pdf",
        "fig8_bridge_validation.pdf",
    }
    present_figures = {path.name for path in (ROOT / "figures").glob("*.pdf")}
    require(expected_figures <= present_figures, "one or more vector figures are missing")
    checks["figures"] = {"expected": len(expected_figures), "present": len(present_figures)}

    refine = pd.read_csv(DATA / "model_i_equilibrium_grid_refinement.csv")
    require(refine.groupby("regime").classification.nunique().max() == 1, "grid class changed on refinement")
    solver = pd.read_csv(DATA / "model_i_solver_crosscheck.csv")
    require(solver.max_absolute_state_difference.max() < 1e-3, "Euler/RK state discrepancy exceeds tolerance")
    require(solver.max_absolute_G_difference.max() < 2e-4, "Euler/RK G discrepancy exceeds tolerance")
    interventions = pd.read_csv(DATA / "model_i_interventions.csv").set_index("intervention")
    require(interventions.loc["reference-restore M+L", "endpoint_state_deficit"] < 0.30,
            "reference restoration no longer closes the state deficit")
    require(abs(interventions.loc["zero M+L once", "absolute_area"]
                - interventions.loc["none", "absolute_area"]) < 0.01,
            "diagnostic one-shot zeroing no longer shows regeneration")
    checks["model_i"] = {
        "grid_refinement_consistent": True,
        "max_solver_state_difference": float(solver.max_absolute_state_difference.max()),
        "reference_restore_state_deficit": float(interventions.loc["reference-restore M+L", "endpoint_state_deficit"]),
    }

    network = pd.read_csv(DATA / "adaptive_network_stress_metrics.csv")
    require(set(network.metric) == {"Q", "r_x", "phi", "P"}, "network metric notation regressed")
    bridge = pd.read_csv(DATA / "adaptive_network_bridge_drift_validation.csv")
    trajectories = pd.read_csv(DATA / "adaptive_network_stress_trajectories.csv")
    starts = trajectories.sort_values(["seed", "regime", "level"]).groupby(["seed", "regime"]).first()
    require((starts.cross_edges_before_rewiring_block == 96).all(), "initial cross-edge target is not exact")
    nonzero = bridge[bridge.interior_predicted_drift_per_update != 0].copy()
    nonzero["acceptance"] = np.where(
        nonzero.interior_predicted_drift_per_update > 0,
        nonzero.reverse_acceptance_rate,
        nonzero.homophilic_acceptance_rate,
    )
    feasible = nonzero[nonzero.acceptance >= 0.95]
    correlation = float(feasible[["mean_observed_drift_per_update", "interior_predicted_drift_per_update"]].corr().iloc[0, 1])
    mae = float((feasible.mean_observed_drift_per_update - feasible.interior_predicted_drift_per_update).abs().mean())
    require(len(feasible) >= 20 and correlation > 0.99 and mae < 0.06, "bridge interior validation failed")
    checks["model_ii"] = {"feasible_blocks": len(feasible), "drift_correlation": correlation, "drift_mae": mae}

    power = json.loads((RESULTS / "power_analysis_summary.json").read_text(encoding="utf-8"))
    require(power["outer_replicates_per_scenario"] == 1000, "outer power budget mismatch")
    require(power["randomization_reassignments_per_dataset"] == 499, "reassignment budget mismatch")
    history = pd.read_csv(DATA / "power_history_grid.csv")
    reset = pd.read_csv(DATA / "power_reset_grid.csv")
    weak_null = pd.read_csv(DATA / "power_weak_null_heterogeneity.csv")
    require(min(history.type1.min(), reset.type1.min()) >= 0.02, "power screen is severely conservative")
    require(max(history.type1.max(), reset.type1.max()) <= 0.08, "power screen inflates type I error")
    require(max(history.mcse.max(), reset.mcse.max()) < 0.017, "power MCSE exceeds declared bound")
    require((weak_null.finite_population_average_effect == 0).all(), "weak-null diagnostic average is not zero")
    require((weak_null.sharp_null == False).all(), "weak-null diagnostic does not falsify the sharp null")
    require(weak_null.rejection_rate.min() >= 0.02, "weak-null diagnostic is severely conservative")
    require(weak_null.rejection_rate.max() <= 0.08, "weak-null diagnostic inflates type I error")
    checks["power"] = {
        "outer_datasets": 1000, "reassignments": 499,
        "type1_range": [float(min(history.type1.min(), reset.type1.min())), float(max(history.type1.max(), reset.type1.max()))],
        "max_mcse": float(max(history.mcse.max(), reset.mcse.max())),
        "heterogeneous_weak_null_rejection_range": [
            float(weak_null.rejection_rate.min()), float(weak_null.rejection_rate.max())
        ],
    }

    manifest_path = ROOT / "release_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            require(path.is_file(), f"manifest file missing: {item['path']}")
            require(sha256(path) == item["sha256"], f"manifest hash mismatch: {item['path']}")
        checks["release_manifest"] = {"files_verified": len(manifest["files"])}

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "archive_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
