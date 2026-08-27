"""Build a Journal of Complex Networks review submission.

JCN requests the standard article class with a 126 mm by 195 mm text block.
The initial submission is a PDF; this directory also retains a self-contained
source file and local vector figures for transparent review and later transfer.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
FIGURES = ROOT / "figures"
SUBMISSION = ROOT / "submission-jcn"

FIGURE_MAP = {
    "fig1_memory_ledger.pdf": "Fig1.pdf",
    "fig2_causal_dag.pdf": "Fig2.pdf",
    "fig2_closed_loop.pdf": "Fig3.pdf",
    "fig3_rate_ablation.pdf": "Fig4.pdf",
    "fig4_sensitivity.pdf": "Fig5.pdf",
    "fig5_bifurcation.pdf": "Fig6.pdf",
    "fig6_adaptive_network.pdf": "Fig7.pdf",
    "fig8_bridge_validation.pdf": "Fig8.pdf",
    "fig5_experimental_design.pdf": "Fig9.pdf",
}


def braced_argument(source: str, command: str) -> str:
    start = source.index(command) + len(command)
    if source[start] != "{":
        raise RuntimeError(f"Expected braced argument after {command}")
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise RuntimeError(f"Unclosed braced argument after {command}")


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    source = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    abstract = braced_argument(source, r"\abstract")
    keywords = braced_argument(source, r"\keywords")
    body = source[source.index(r"\section{Introduction}") :]
    references = (MANUSCRIPT / "references.tex").read_text(encoding="utf-8").strip()
    if body.count(r"\input{references.tex}") != 1:
        raise RuntimeError("Expected exactly one references.tex input marker")
    body = body.replace(r"\input{references.tex}", references)

    for original, flattened in FIGURE_MAP.items():
        old_path = f"../figures/{original}"
        if old_path not in body:
            raise RuntimeError(f"Figure path not found in manuscript: {old_path}")
        body = body.replace(old_path, flattened)
        shutil.copy2(FIGURES / original, SUBMISSION / flattened)

    body = body.replace(
        r"\section*{Statements and Declarations}" + "\n\n" + r"\inlinehead{Funding.}",
        r"\section*{Funding}",
    )
    body = body.replace(r"\inlinehead{Acknowledgements.}", r"\section*{Acknowledgements}")

    preamble = r"""% Journal of Complex Networks initial-review manuscript.
% Standard article.cls; text block follows the journal's 30 x 46 pica rule.
\documentclass[10pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[textwidth=126mm,textheight=195mm,centering]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts,amsthm,bm}
\usepackage{booktabs,tabularx,array}
\usepackage{xcolor}
\usepackage{url}
\usepackage{microtype}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[toc]{appendix}
\usepackage[section]{placeins}
\usepackage{hyperref}

\hypersetup{
  pdftitle={Testing Algorithmically Mediated Social Hysteresis: Closed-Loop Causal Tests and Reversal Control},
  pdfauthor={Simone D'Agnano},
  pdfsubject={Computational social science; adaptive networks; recommender systems; reversal controllability},
  pdfkeywords={social hysteresis, algorithmic mediation, adaptive networks, recommender systems, causal identification, reversal controllability},
  colorlinks=true,
  linkcolor=blue,
  citecolor=blue,
  urlcolor=blue
}

\raggedbottom
\setlength{\emergencystretch}{3em}
\newcommand{\alphamax}{\alpha_{\max}}
\newcommand{\Gup}{G_{\uparrow}}
\newcommand{\Gdown}{G_{\downarrow}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\inlinehead}[1]{\par\smallskip\noindent\textbf{#1}\ }
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\alttext}[1]{\par\smallskip\noindent\textit{Alt text:} #1}

\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}
\renewcommand{\arraystretch}{1.18}
\setlength{\arrayrulewidth}{0.45pt}

\title{Testing Algorithmically Mediated Social Hysteresis:\\
Closed-Loop Causal Tests and Reversal Control}
\author{Simone D'Agnano\\
\small Dipartimento di Scienze e Innovazione Tecnologica (DISIT),\\
\small Universit\`a del Piemonte Orientale, Viale Teresa Michel 11,\\
\small 15121 Alessandria, Italy\\
\small \href{mailto:s.dagnano.research@gmail.com}{s.dagnano.research@gmail.com}\\
\small \href{https://orcid.org/0009-0003-6394-9408}{ORCID 0009-0003-6394-9408}}
\date{}

\begin{document}
\maketitle
"""
    front = (
        "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
        "\\noindent\\textbf{Keywords:} " + keywords + "\n\n"
    )
    output = preamble + front + body
    (SUBMISSION / "submission.tex").write_text(output, encoding="ascii")
    shutil.copy2(MANUSCRIPT / "references.bib", SUBMISSION / "references.bib")
    print(f"Prepared {SUBMISSION}")


if __name__ == "__main__":
    main()
