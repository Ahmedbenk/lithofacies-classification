# Project Log: Lithofacies Classification — SEG + FORCE 2020

A chronological summary of this project, written as a companion to the
raw chat export (if you download that separately via Claude.ai's
Settings > Privacy > Export Data). This captures the *reasoning* behind
each decision, not just the final numbers.

---

## 1. Starting point

Original goal: predict lithology units from well-log data, compare ML
techniques, and build a hybrid method for a paper targeting *Artificial
Intelligence in Geosciences* (KeAi/Elsevier). Started with an uploaded
Python script combining XGBoost and a neural network, plus the SEG
facies dataset (`facies_data.csv`).

**Initial code review found:**
- Data leakage: well identity one-hot encoded as a feature, plus random
  (not well-grouped) train/test splitting
- The "hybrid" stacked XGBoost predictions into a neural net using
  in-sample (not out-of-fold) predictions — a second leakage source
- The well-log plots used randomly generated fake predictions
  (`np.random.choice`), not real model output
- Hardcoded Windows file path

Fixed by moving the whole workflow to Google Colab (with Drive mounting)
and rebuilding the pipeline from scratch.

## 2. Establishing rigorous validation

Settled on leave-one-well-out (LOWO) cross-validation for SEG (8 wells,
fully enumerable) and found the official SEG blind-well protocol
(STUART/CRAWFORD wells, held out from the original 2016 contest) —
retrieved `validation_data_nofacies.csv` and
`blind_stuart_crawford_core_facies.csv` from the contest's GitHub repo.
Established the rule: blind wells get looked at **once**, only after all
tuning is done via cross-validation — never used to guide decisions.

## 3. The hybrid-mechanism saga (SEG dataset)

Tried four different hybrid mechanisms, in this order, each targeting a
specific diagnosed weakness:

1. **NN stacking** (XGBoost out-of-fold probabilities + neural network
   meta-learner). First attempt used full inverse-frequency class
   weighting — collapsed facies 9 recall from 0.789 to 0.224. Second
   attempt used dampened (square-root) weighting + early stopping —
   partially recovered facies 9 (to 0.366) but broke facies 5 (0.157 to
   0.046). **Net result: worse than baseline both times.**
2. **HMM/Viterbi sequence smoothing**, exploiting strong depth-order
   structure. Mixed result — helped some classes, suppressed others via
   majority-class transition bias. **Only tested with a RandomForest
   stand-in, not real XGBoost — flagged as preliminary throughout.**
3. **Cascade binary specialist** (SVM) targeting the facies 8/9
   confusion specifically. Made things worse — the specialist had no new
   information beyond what the base model already used. **Also
   RandomForest stand-in only.**
4. **Depth-window rolling features** (mean/std over a 5-sample window).
   Didn't help the target class (facies 5), slight net accuracy
   decrease. **Also RandomForest stand-in only.**

**Key finding**: all four failed to beat a well-calibrated single
XGBoost. This became a central, honestly-reported negative result rather
than something to hide or keep chasing.

**IMPORTANT UPDATE (August 9 re-validation run)**: Mechanisms #2-4 were
re-run with the real XGBoost production model (replacing the RandomForest
stand-in). Cascade specialist and depth-window features confirmed the
negative result (still worse than baseline). BUT HMM/Viterbi smoothing
flipped: it now BEATS baseline outright (accuracy 0.5068 -> 0.5266,
adjacent-accuracy 0.8651 -> 0.8784), at the cost of further collapsing
facies 1 and 5 recall (0.17 -> 0.14, 0.15 -> 0.09). The manuscript's
diagnosis table entry for facies 8 ("cascade specialist + HMM smoothing
tested, no improvement found") needs revising -- the HMM half of that
claim is no longer accurate. Full output in `results/revalidation_cell8_output.txt`.

**Original caveat (now resolved)**: only mechanism #1 (NN stacking) was
ever run with the real production model (XGBoost + Keras) in the original
Colab session. Mechanisms #2-4 were only tested with a RandomForest
substitute in Claude's sandbox (XGBoost wasn't installed there). The
script `revalidate_hybrids_xgboost.py` was written to close this gap and
has now been run (see update above).

## 4. Pivot: diagnostic framework instead of a universal hybrid

After four failed attempts, agreed on a different approach: diagnose
*why* each underperforming class struggles, then apply a remedy matched
to that specific diagnosis, rather than forcing one hybrid architecture
to fix everything. Four diagnostic categories were defined:

1. Adjacency confusion (e.g., SEG facies 8/9)
2. Missing discriminating feature (e.g., FORCE's Chalk/Anhydrite needing PEF)
3. Structural data scarcity (e.g., FORCE's Basement, confined to 1 well)
4. Genuinely ambiguous class (e.g., FORCE's Sandstone/Shale)

## 5. Extending to the FORCE 2020 dataset

Moved to a second, much larger dataset (1,170,511 rows, 98 wells, 12
classes) to test whether the calibration + diagnosis approach generalizes.
Went through a real detour finding the correct data source — two wrong
downloads (seismic/OpendTect project files from a look-alike competition)
before landing on the correct GitHub repo
(`bolgebrygg/Force-2020-Machine-Learning-competition`).

**Key results:**
- Full inverse-frequency weighting caused a precision collapse (Dolomite
  precision 0.005) — dampened weighting fixed this (accuracy 0.5409 to
  0.6686), confirming the same calibration fix that helped SEG.
- Feature-availability ablation: adding PEF measurably helped
  Chalk/Anhydrite/Limestone (a real, testable, confirmed hypothesis).
  Adding NPHI (despite being the "textbook" dolomite indicator) did NOT
  help Dolomite — a falsified hypothesis, reported honestly rather than
  omitted.
- Discovered that 4 classes (Halite, Anhydrite, Dolomite, Basement) are
  concentrated in very few wells (3, 6, 37, and 1 out of 98
  respectively) — Basement specifically cannot be fairly cross-validated
  under any well-based split.

## 6. The generalization-gap discovery

Running both datasets' official blind-well tests (one honest look each,
per the established rule) surfaced the project's most important finding:
**classes concentrated in few wells can look fine under cross-validation
but collapse on genuinely unseen wells.** Chalk: CV recall 0.307 → blind
recall 0.027. Anhydrite: 0.377 → 0.000. This wasn't visible from either
dataset's cross-validation alone — it only showed up by comparing CV
results against true blind-well performance.

**Honesty check performed**: searched the literature to verify how novel
this finding actually was. Found it's a specific instance of a
well-documented general ML phenomenon (dangers of cross-validation,
Kohavi 1995; Rao/Fung/Rosales 2008), not a brand-new discovery — the
paper's contribution is the specific, quantified diagnosis via
fold-presence tracking within lithofacies classification, not the
underlying principle itself.

## 7. Building the manuscript

Structured the paper to match *Artificial Intelligence in Geosciences*'
required format (Title Page, Abstract <250 words, Keywords, Introduction,
Material and Methods, Theory, Results, Discussion, Conclusions,
Declarations, References with numbered bracket citations).

**A citation-accuracy correction worth noting**: an early citation
(arXiv:2204.14224) was mis-attributed — the ID was real but pointed to a
different paper than described (GAN-based core image classification, not
well-log cross-validation). Caught during final review, removed, and
replaced with a verified, genuinely relevant citation (Halotel, Demyanov
& Gardiner, 2020, *Mathematical Geosciences*).

Built 6 figures (confusion matrices, feature-ablation comparison,
CV-vs-blind recall chart, hybrid-mechanism comparison, and a Techlog-style
well-log comparison plot for STUART), verified pixel-by-pixel for correct
depth-axis orientation before shipping (a subtle bug risk when mixing
line-plot and image-based tracks on a shared, inverted axis).

## 8. What's still open (see paper_FINAL.md's "Document status" section)

1. ~S~Re-run `revalidate_hybrids_xgboost.py` in Colab - DONE Aug 9, see
   update in section 3 above. Manuscript diagnosis table still needs
   updating to reflect the HMM result.
2. Regenerate Figure 2 (STUART log plot) with real predictions - currently
   uses a RandomForest stand-in for the pixel-level pattern (the
   confusion-matrix numbers elsewhere in the paper are already real).
3. Confirm author names for reference [9] (Mathematical Geosciences
   FORCE/Geolink benchmark paper).
4. Fill in Title Page and Declarations placeholders (affiliation,
   funding, conflict of interest, code availability).

---

## File guide (this archive)

- `final_deliverables/paper_FINAL.md` — the current, most complete draft
- `figures/` — all 6 numbered figures, matching the final draft
- `code/` — all pipeline and figure-generation scripts
- `results/` — August 9 re-validation run output (real XGBoost, all four
  mechanisms)
- `superseded_drafts/` — earlier draft versions, kept for reference only

