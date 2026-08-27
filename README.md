# Testing Algorithmically Mediated Social Hysteresis

Reproducibility archive for:

**Simone D'Agnano — _Testing Algorithmically Mediated Social Hysteresis: Closed-Loop Causal Tests and Reversal Control_**

Repository:  
https://github.com/sdagnano/algorithmically-mediated-social-hysteresis

## Overview

This repository contains the manuscript, code, derived data, figures, literature audit, and computational checks associated with the paper.

The paper treats social hysteresis, recommender feedback, adaptive-network memory, and withdrawal asymmetry as prior ideas. Its contribution is methodological: it develops a causal framework for testing whether different randomized algorithmic policy histories produce different outcomes at matched current policy, whether a prespecified finite-rate relaxation class can explain the resulting branch gap, and which admissible interventions restore the resulting state.

The framework combines:

- complete-history causal estimands on randomized graded paths;
- finite-record observational-equivalence results;
- step-and-dwell and continuous Wasserstein relaxation bounds;
- total-system and candidate-standardized estimands;
- reset-sensitive and channel-attributable intervention targets;
- paired path/state reversal deficits;
- cost-constrained reversal policies;
- adaptive-network structural-recovery analysis;
- and a gatekept cluster-randomized experimental design.

**All numerical results are synthetic and uncalibrated. They are mechanism and design stress tests, not evidence that any real platform or population exhibits quasistatic hysteresis.**

## Main computational studies

### Model I

Model I is a four-state normal form containing:

- collective state \(G\);
- recommender-profile memory \(M\);
- topological memory \(L\);
- human or normative memory \(C\).

The archive includes:

- full-system equilibrium and Jacobian stability analysis;
- rate sweeps;
- mechanism ablations;
- one-shot diagnostic zeroing;
- contemporaneous-reference platform restoration;
- human-recovery facilitation;
- logistic/probit link comparison;
- process-noise stress tests;
- local parameter sensitivity;
- 1,000 all-parameter Latin-hypercube draws;
- and an independent Dormand–Prince solver audit.

### Model II

Model II is an adaptive network with continuous node states and stochastic degree-preserving rewiring.

The archive includes:

- 32 independent initializations;
- reversible and irreversible rewiring regimes;
- multiple driving rates;
- no-rewiring controls;
- exact graph restoration;
- bridge-count drift diagnostics;
- proposal acceptance rates;
- finite-horizon Poisson repair calculations;
- loop areas and endpoint gaps.

Model II represents the **topological memory channel only**. It does not contain a recommender-learning or content-production layer.

### Experimental-design feasibility

The proposed empirical program contains five stages:

1. blinded pilot;
2. randomized history test;
3. relaxation-class test;
4. platform-state restoration;
5. replication and minor loops.

The power module recomputes the studentized maximum statistic under dataset-level randomization for every simulated dataset.

A separate audit also tests heterogeneous treatment effects with an exactly zero finite-population average effect, so that the Fisher sharp null is false while the weak null is true.

## Quick start

Requirements:

- Python 3.11+
- packages in `requirements.txt`
- MiKTeX or TeX Live for PDF compilation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the archive:

```bash
python code/verify_archive.py
```

On Windows, regenerate the complete archive:

```powershell
./build.ps1
```

## Manual reproduction

The main computational pipeline is:

```text
python code/simulate.py
python code/model_i_stress.py
python code/adaptive_network_stress.py
python code/design_power.py
python code/verify_archive.py
```

The manuscript/submission rendering can then be regenerated with the corresponding preparation and LaTeX build scripts.

## Repository structure

```text
manuscript/       authoritative manuscript source
code/             simulation, analysis, audit, and build scripts
data/derived/     regenerated numerical outputs
figures/          vector manuscript figures
literature/       bibliography, DOI audit, and search documentation
results/          machine-readable result summaries
submission-jcn/   generated review-format rendering
submission/       alternate generated rendering
```

`manuscript/` is the **authoritative scientific source**. Submission directories are generated renderings and do not represent different scientific versions of the paper.

## Verification

`code/verify_archive.py` checks:

- required files;
- citation closure;
- bibliography consistency;
- key numerical invariants;
- model-result consistency;
- adaptive-network invariants;
- power-analysis summaries;
- and stale-claim regressions.

`release_manifest.json` records SHA-256 hashes for the archived release files.

## Literature audit

The bibliography is a structured cross-disciplinary audit, not a PRISMA systematic review or a claim of literal exhaustiveness.

The final frozen set contains **107 DOI-bearing records** covering social hysteresis, adaptive networks, algorithmic opinion dynamics, recommender feedback, platform experiments, classical hysteresis identification and control, causal inference, interference, and closed-loop recommender control.

The literature status is frozen at **26 August 2026**.

## Reproducibility limitations

Reproducing the archive does not establish empirical validity of the synthetic models.

In particular:

- neither model is calibrated to a real platform;
- Model I is a reduced-order normal form;
- Model II is a stylized adaptive network;
- the power analysis uses a planning data-generating process;
- the relaxation-class test requires externally justified regularity bounds;
- finite-rate experiments do not nonparametrically prove a quasistatic limit;
- and exact platform-state restoration is an experimental idealization.

## Citation

Until an arXiv identifier or Version of Record is available, please cite:

```text
D'Agnano, S. (2026).
Testing Algorithmically Mediated Social Hysteresis:
Closed-Loop Causal Tests and Reversal Control.
Preprint.
```

Machine-readable citation metadata are provided in `CITATION.cff`.

## Ethics

The reported work is theoretical and computational and uses no human participants, identifiable personal data, or proprietary platform data.

Any future experiment described in the manuscript would require prospective ethics review, appropriate consent, exposure limits, harm monitoring, participant exit, debriefing, and preregistered stopping rules.

## Generative AI disclosure

OpenAI Codex was used during manuscript development for literature-discovery assistance, code generation, figure drafting, and language generation.

The author remains accountable for all scientific claims, citations, computational results, and released materials.

## License

Code licensing is specified in `LICENSE-CODE`.

Manuscript and figure licensing is specified in `LICENSE-MANUSCRIPT`.

## Author

**Simone D'Agnano**  
Dipartimento di Scienze e Innovazione Tecnologica (DISIT)  
Università del Piemonte Orientale, Italy

ORCID: `0009-0003-6394-9408`  
Email: `s.dagnano.research@gmail.com`

Repository:  
https://github.com/sdagnano/algorithmically-mediated-social-hysteresis