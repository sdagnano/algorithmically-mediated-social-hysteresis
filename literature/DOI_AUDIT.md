# DOI and closest-precedent audit

Audit date: 26 August 2026.

## Scope and method

The original frozen corpus contained 87 DOI-bearing records. Each record was compared with DOI-registry metadata for title, ordered author list, bibliographic year, container, volume, issue, and page or article number. Crossref supplied 85 records and DataCite supplied the two `10.48550/arXiv` records. All 87 DOIs resolved. Publisher or official index pages were consulted where an online-first date differed from the issue citation.

The check corrected 19 bibliographic metadata fields affecting 15 existing records: six high-priority year, venue, or version issues and thirteen medium-priority article-number or issue fields. The discrepancy log also records one conceptual repositioning (Zeng et al. as the closest direct predecessor). Chan plus nineteen subsequent additions bring the complete log to 40 documented actions. The audit found no unresolved DOI, duplicate work, wrong DOI-to-title association, title mismatch, or omitted author after harmless differences in capitalization, punctuation, diacritics, and venue abbreviation were normalized.

Johnny Chan's 2026 Version of Record was then verified through Crossref and Wiley and added as record 88. Reviewer-driven updates added nineteen further records covering Bak-Coleman's closest causal critique, classical hysteresis identification/control, algorithmic drift, network-aware recommender feedback, adaptive-network topological hysteresis, closed-loop recommender control, and clustered or saturation-design randomization inference. These records were checked against publisher or official repository pages. Davidson and Ye's arXiv entry was subsequently replaced by the 2026 IEEE Version of Record (DOI 10.1109/TCSS.2026.3715162). The final 107-record corpus contains 100 Crossref-resolved and seven DataCite-resolved records.

The machine-readable pre-correction findings are in `doi_metadata_discrepancies.csv`. Corrections are encoded in `code/build_bibliography.py`, and the regenerated `curated_bibliography.csv`, `references.tex`, and `references.bib` are the authoritative post-correction outputs.

## Closest direct predecessor

An Zeng, Chi Ho Yeung, Matúš Medo, and Yi-Cheng Zhang (2015), “Modeling mutual feedback between users and recommender systems,” *Journal of Statistical Mechanics: Theory and Experiment* 2015(7), P07020. DOI: `10.1088/1742-5468/2015/07/P07020`.

Zeng et al. explicitly report a hysteresis effect in a coevolving user-item/recommender model. A sweep of recommendation bias changes the stationary Gini coefficient of item popularity, and reversing the bias does not fully restore diversity. The manuscript therefore names this paper as the closest direct recommender-hysteresis predecessor. The remaining contribution is not the first algorithmic loop; it is causal identification of human and adaptive-network states at matched current mediation, including rate tests, reset localization, and reversal controllability.

Primary author manuscript: <https://arxiv.org/abs/1508.01672>

## Closest 2026 conceptual predecessor

Johnny Chan (2026), “Where the Brake Slips: A Feedback View of the Micro-Drama Platform,” *Systems Research and Behavioral Science*, Early View, 1-7. DOI: `10.1002/sres.70126`.

Chan proposes user-, platform-, and market-level hysteresis and suggests longitudinal observation, imposed breaks, payment friction, and recommender resets. The paper labels the proposition a conjecture and uses qualitative causal-loop analysis rather than a quantitative closed-path estimand. It is therefore cited as the closest multilevel conceptual predecessor, not as a duplicate of the identification framework.

Version of Record: <https://onlinelibrary.wiley.com/doi/10.1002/sres.70126>

## Confirmed high-priority corrections

- Huszár et al., DOI `10.1073/pnas.2025334119`, is cited as *Proceedings of the National Academy of Sciences* 119(1), e2025334119 (2022), rather than as an arXiv record or a 2021 issue citation.
- Oliveira, Ferraz de Arruda, and Moreno, DOI `10.1093/pnasnexus/pgaf402`, is cited as *PNAS Nexus* 5(1), pgaf402 (2026); its Version of Record first appeared online on 30 December 2025.
- Arruda et al., DOI `10.1016/j.ins.2021.12.069`, uses the 2022 issue year.
- Bojinov et al., DOI `10.1287/mnsc.2022.4583`, uses the 2023 issue year.
- Sinha, Gleich, and Ramani, DOI `10.48550/arXiv.1703.01049`, is represented consistently as the 2017 arXiv version rather than mixing a 2016 proceedings citation with a 2017 arXiv DOI.
- Eleven records now include their registry-supplied article number.

This audit supports bibliographic accuracy as of the audit date; it is not a guarantee against later publisher corrections or metadata changes.

## Targeted updates after the frozen search

The updates deliberately change the paper's novelty boundary. Bak-Coleman et al. are cited as the closest conceptual-causal predecessor for treatment--state non-equivalence; their arXiv record is cited as 2025, version 3 revised 23 April 2026. Bak-Coleman's *Nature Reviews Psychology* item is represented as a 2025 Correspondence in volume 4, page 615. Eight classical sources delimit what is inherited from Preisach, Duhem, return-point memory, identification, and inverse-control theory. Coppolillo et al. is cited in *Information Processing \& Management* 62(4), 104125 (2025). Chandrasekaran et al. is cited in *IEEE Transactions on Automatic Control* 71(3), 1708--1723 (2026), with its 2025 DOI retained.

The final update adds Asikainen et al.'s explicit topological-memory/hysteresis result, Davidson and Ye's coupled recommender--opinion loop (updated to its 2026 IEEE Version of Record), three 2026 closed-loop recommender control papers, Wu and Ding's weak-null randomization framework, and Liu, Shaikh, and Zhong's saturation-design randomization tests. These works narrow the residual contribution to matched-history identification, rate-class falsification, and reversal control rather than a first claim about social or algorithmic hysteresis.
