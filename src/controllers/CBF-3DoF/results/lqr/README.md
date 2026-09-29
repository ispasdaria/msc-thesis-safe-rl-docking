# LQR with CBF results

This directory contains paired evaluations of the repository LQR controllers with and without the hard CBF-QP filter.

- `closing_dt0p1_full_cbf_family_breakdown_seeds_0_49/` is the retained 0.1 s closing evaluation with per-constraint intervention fields.
- `final_dt1p0_full_cbf_family_breakdown_seeds_0_49/` is the retained 1 s final-approach evaluation with the same family breakdown.
- `closing_dt0p1_no_cbf_native_initialization_seeds_0_49/` evaluates the unfiltered closing LQR from its native testing distribution.
- `closing_dt0p1_no_cbf_p3_initialization_seeds_0_49/` is the distinct P3-initialized comparison.
- `comparison_figures/` contains retained LQR and CBF figures and their selected trajectory histories.

Reduced tables fully superseded by the family-breakdown results and one byte-identical unfiltered duplicate were removed.
