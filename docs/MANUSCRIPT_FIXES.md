# Minimal changes to the 28-page manuscript before journal submission

These are **editorial directions**, not claims that source LaTeX has already been updated.

1. **Appendix F.3 / Code and data availability**: Replace the broad GitHub organization homepage `https://github.com/SequentialDecisionResearch` with the exact project URL `https://github.com/SequentialDecisionResearch/contextual-selective-ope`. After pushing the final files, record the final repository commit hash or release tag. Be candid that complete historic checkpoints and original OBD logs are not bundled.
2. **Section 3.4**: Repair the missing closing quote: `as the “Local Gate”; no additional gate or learned score is fitted.`
3. **Section 3.4 and Section 6**: Keep the proposition conditional on a **valid one-sided** lower confidence bound. Distinguish one-sided lower-bound calibration from two-sided interval coverage statistics in existing experiments. Do **not** claim the nominal 95% level is achieved in the poor-overlap Semi-Synthetic data.
4. **OBD (Sec. 5.6, abstract/conclusion)**: Describe independent on-policy logs as an external reference for evidence-dependent admission, not oracle per-group causal truth. In the forward direction there are no clearly harmful reference groups. State that real-world harmful exposure avoidance is not directly demonstrated in this forward comparison.
5. **Methods / Appendix F**: Include the revision identifiers and actual run protocols from the three uploaded manifests; explain that the current clean future runner is a **different** execution configuration from the completed historic formal runs and that Synthetic S6 was not in the recovered result.
6. **End matter / policy**: Add truthful author contribution, funding, competing interests, dataset licenses/availability and generative-AI assistance disclosures in compliance with the selected journal; authors are responsible for validating original drafts. Do not claim no AI use if any occurred.
7. **Suggested, not necessary for a two-day deadline**: Compare LCB gate vs local point-estimate gate if row-level frozen CSVs permit a cheap secondary analysis, but do not force new Monte Carlo runs. If unavailable, discuss as a limitation.

No scientific result should be silently changed to fit a preferred story. If a key old-vs-final discrepancy cannot be resolved from saved outputs, mark it as unresolved and pause submission.
