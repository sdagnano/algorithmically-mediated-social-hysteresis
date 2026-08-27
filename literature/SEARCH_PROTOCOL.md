# Structured literature audit

Audit date: 26 August 2026.

## Scope

The objective was conceptual and design saturation across adjacent fields needed to evaluate the narrow claim of algorithmically mediated social hysteresis. The audit was not registered as a systematic review and does not claim literal exhaustiveness.

## Query families

OpenAlex title/abstract/metadata search used the following nine families, with 60 results requested per family:

1. `adaptive social network opinion dynamics hysteresis`
2. `social hysteresis opinion dynamics`
3. `algorithmic personalization opinion dynamics social network`
4. `recommender systems feedback loops homogenization`
5. `algorithmic recommendation political polarization social media`
6. `feed algorithm causal experiment social media`
7. `adaptive network homophily rewiring bounded confidence`
8. `echo chambers recommender systems network dynamics`
9. `path dependence algorithmic curation social media`

The searches yielded 876 unique candidates after DOI/OpenAlex-ID deduplication. Backward chaining from ten anchor works resolved 461 cited-work identifiers.

## Anchor works

- `10.1016/j.physa.2020.125588`
- `10.1038/s41598-019-43830-2`
- `10.1038/s41586-026-10098-2`
- `10.1177/17456916231195361`
- `10.1016/j.eswa.2025.126851`
- `10.1016/j.chaos.2025.117660`
- `10.1098/rsif.2007.1229`
- `10.1093/comnet/cnac055`
- `10.1145/3240323.3240370`
- `10.1073/pnas.2313377121`

## Inclusion logic

A work entered the final set if it directly supported at least one of the following roles:

- defines or demonstrates social hysteresis, tipping, or branch dependence;
- supplies an adaptive-network or topology-memory mechanism;
- models algorithmic filtering, recommendation, or coupled opinion dynamics;
- establishes recommender data-algorithm feedback or long-run distortion;
- provides causal or audit evidence about feed algorithms and collective outcomes;
- supplies experimental methods for carryover, switchbacks, dynamics, or network interference;
- reviews a directly relevant empirical or modeling field.

Works were excluded when the title/abstract connection was only lexical, when the system had no social or mediation relevance, when a closer primary source was available, or when the claim was duplicated without adding mechanism or identification value.

## Outputs and validation

`selected_sources.csv` contains 107 inclusion decisions and assigned roles. `curated_bibliography.csv` contains normalized metadata; every selected record has a DOI. `references.tex` and `references.bib` mirror the same corpus. `DOI_AUDIT.md` and `doi_metadata_discrepancies.csv` document the field-level audit. All special characters in the submission bibliography are TeX-encoded.

After the frozen broad search, reviewer-driven targeted updates added nineteen records: Bak-Coleman et al.'s closest conceptual-causal treatment, Bak-Coleman's randomized-trial critique, eight classical hysteresis identification/control sources, algorithmic drift, network-aware feedback optimization, topological hysteresis in adaptive social networks, five closed-loop recommender/opinion-control records, and two randomization-inference sources. These updates change the novelty boundary rather than claiming that a fixed query set is eternally exhaustive.

The final distribution is:

- platform evidence: 19;
- recommender feedback: 18;
- adaptive networks: 16;
- social hysteresis: 21;
- algorithmic opinion dynamics: 15;
- reviews: 9;
- causal design: 9.

The audit is intentionally broader than the citation list needed for a narrow model paper. Every selected record is nevertheless cited in the manuscript so that the bibliography contains no decorative or orphan entries.

## DOI metadata verification

The original 87 records were checked against DOI-registry metadata field by field: title, ordered author list, bibliographic year, container, volume, issue, and page or article number. Crossref supplied 85 records and DataCite supplied two arXiv records; all resolved. The audit corrected 19 bibliographic metadata fields affecting 15 existing records. Johnny Chan's published 2026 Research Note was verified through Crossref and Wiley and added as record 88. The nineteen-record targeted update was then checked against publisher or official repository records. Davidson and Ye's arXiv record was later replaced by the 2026 IEEE Version of Record, yielding 107 sources: 100 Crossref records and seven DataCite arXiv records. No duplicate DOI or key is present.
