"""Build a flattened Springer Nature submission directory.

The journal submission system expects one TeX document and local figure paths.
This script inlines the audited bibliography and copies the official class,
bibliography style, and vector figures into ``submission/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
FIGURES = ROOT / "figures"
SUBMISSION = ROOT / "submission"

FIGURE_MAP = {
    "fig1_memory_ledger.pdf": "Fig1.pdf",
    "fig2_causal_dag.pdf": "Fig2.pdf",
    "fig2_closed_loop.pdf": "Fig3.pdf",
    "fig3_rate_ablation.pdf": "Fig4.pdf",
    "fig4_sensitivity.pdf": "Fig5.pdf",
    "fig6_adaptive_network.pdf": "Fig6.pdf",
    "fig5_experimental_design.pdf": "Fig7.pdf",
}


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    source = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    references = (MANUSCRIPT / "references.tex").read_text(encoding="utf-8").strip()

    marker = r"\input{references.tex}"
    if source.count(marker) != 1:
        raise RuntimeError("Expected exactly one references.tex input marker")
    source = source.replace(marker, references)

    for original, flattened in FIGURE_MAP.items():
        old_path = f"../figures/{original}"
        if old_path not in source:
            raise RuntimeError(f"Figure path not found in manuscript: {old_path}")
        source = source.replace(old_path, flattened)
        shutil.copy2(FIGURES / original, SUBMISSION / flattened)

    (SUBMISSION / "submission.tex").write_text(source, encoding="ascii")
    shutil.copy2(MANUSCRIPT / "sn-jnl.cls", SUBMISSION / "sn-jnl.cls")
    shutil.copy2(MANUSCRIPT / "sn-mathphys-num.bst", SUBMISSION / "sn-mathphys-num.bst")
    shutil.copy2(MANUSCRIPT / "references.bib", SUBMISSION / "references.bib")

    print(f"Prepared {SUBMISSION}")


if __name__ == "__main__":
    main()
