# Geolink PEF-inclusion test -- Dolomite / Calcareous Cement

Targeted remedy test per the diagnostic mapping (geolink_diagnostic_mapping_2026-08-30.md),
same methodology as FORCE's PEF/NPHI feature-inclusion tests (paper Section 4.2).
Run on Kaggle. Restricted to the subset of wells that actually have PEF
(85/223 wells, 439,245 rows after min-50 filter -- fewer than the full
curve-survey's 91-well PEF count because this also requires all 5 baseline
curves simultaneously), comparing baseline (5 curves) vs. baseline+PEF (6
curves) on the exact same well subset, so only the feature changes.

**Overall accuracy: 0.3944 (no PEF) -> 0.3997 (with PEF)** -- a small net
positive across all classes pooled, but the two target classes tell a
different story.

## Result: both Category 2 hypotheses are FALSIFIED

| Class | Support | F1 no-PEF | F1 with-PEF | Delta |
|---|---|---|---|---|
| Dolomite | 2,679 | 0.000 | 0.000 | 0.000 |
| Calcareous Cement | 9,474 | 0.071 | 0.046 | -0.025 |

- **Dolomite: completely unaffected.** Exactly zero recall/F1 with or
  without PEF -- this is not a missing-discriminating-feature problem.
  Rules out Category 2 entirely for this class.
- **Calcareous Cement: actively hurt by PEF** (F1 drops ~35% relative).
  Adding PEF didn't just fail to help -- it made the model worse, the same
  kind of finding as FORCE's NPHI test degrading classes PEF had already
  helped (Section 4.3). Also rules out Category 2 for this class.

This directly parallels the paper's existing finding that feature-
availability diagnosis is falsifiable, not just confirmable (NPHI failing
to fix FORCE's Dolomite despite being a textbook indicator). Worth citing
alongside that result in Discussion as a second, independent instance of
the same pattern.

## Revised diagnosis

Both reassigned from "Category 2 pending" to **Category 4 (genuine
ambiguity)**, with a geological rationale that fits the pattern:

- **Dolomite** commonly forms through diagenetic alteration of limestone
  (dolomitization), which produces a genuine textural/compositional
  continuum rather than a sharp boundary -- consistent with confusion
  against Limestone/Argillaceous Limestone/Marlstone (already flagged as
  Category 1 adjacency classes) rather than a missing-log problem.
- **Calcareous Cement** describes a cementation style overprinted on a host
  rock rather than a primary lithology -- plausibly blends spectrally with
  whatever rock it's cementing (Sandstone, Limestone), another case where
  the class boundary reflects a real geological gradient, not absent data.

Both should be moved into the Category 4 group alongside the silt/sand
grain-size continuum in the diagnostic mapping doc.

## Full per-class comparison

See attached CSV-equivalent table in the run log for all 19 classes present
in the PEF-well subset; no other class showed a delta larger than +/-0.04,
so the effect is genuinely concentrated on Limestone (+0.040) and Chalk
(+0.034) -- both plausible real PEF beneficiaries -- rather than a general
lift across the board.

## Correction after counting wells per class

Counting the wells that contain each class changes the Dolomite assignment above. Dolomite occurs in only 8 wells, and 83% of its samples come from one of them, so structural scarcity (category 3) explains its failure better than ambiguity. Calcareous Cement occurs in 164 wells, with no well holding more than 16% of its samples, so it stays under suspected ambiguity (category 4). In both cases the PEF test did not support a missing-feature explanation. Well counts per class are in Table 3 of the paper.
