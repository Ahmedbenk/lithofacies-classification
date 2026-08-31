# Geolink Phase 1 baseline -- FULL dataset (via Kaggle)

Ran on Kaggle (kaggle.com/code/ahmadbenkhial/notebookafb150abad) after the
sandbox we'd been using twice OOM-killed on this dataset at 1GB RAM. Full
cleaned dataset, no per-class cap: 1,184,461 rows, 181 wells, 22 classes
(min-50-sample threshold), 10-fold StratifiedGroupKFold, calibrated single
XGBoost with dampened (sqrt) class weighting -- same methodology as SEG and
FORCE.

**Overall accuracy: 0.3877**

| Lithology | Code | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Silt | 5 | 0.456 | 0.349 | 0.396 | 288,792 |
| Shaly Silt | 6 | 0.479 | 0.466 | 0.473 | 276,702 |
| Marlstone | 13 | 0.413 | 0.490 | 0.448 | 121,200 |
| Silty Shale | 7 | 0.400 | 0.469 | 0.432 | 92,510 |
| Silty Sand | 2 | 0.321 | 0.340 | 0.330 | 83,252 |
| Cross Bedded Sst | 3 | 0.327 | 0.379 | 0.351 | 75,559 |
| Sandy Silt | 4 | 0.101 | 0.057 | 0.073 | 45,601 |
| Limestone | 11 | 0.501 | 0.525 | 0.513 | 44,152 |
| Argillaceous Limestone | 12 | 0.226 | 0.281 | 0.251 | 39,655 |
| Sandstone | 1 | 0.299 | 0.307 | 0.303 | 36,550 |
| Calcareous Cement | 16 | 0.071 | 0.056 | 0.063 | 26,222 |
| Cinerite | 19 | 0.187 | 0.406 | 0.256 | 15,408 |
| Chalk | 9 | 0.260 | 0.231 | 0.245 | 13,530 |
| Porous Limestone | 10 | 0.412 | 0.438 | 0.425 | 10,709 |
| Coal | 18 | 0.281 | 0.660 | 0.394 | 5,113 |
| Dolomite | 15 | 0.007 | 0.014 | 0.009 | 3,075 |
| Anhydrite | 22 | 0.577 | 0.748 | 0.651 | 2,691 |
| Halite | 21 | 0.471 | 0.224 | 0.303 | 1,659 |
| Conglomerate | 17 | 0.000 | 0.000 | 0.000 | 1,013 |
| Gap | 26 | 0.003 | 0.008 | 0.004 | 600 |
| Carnallite | 33 | 0.000 | 0.000 | 0.000 | 414 |
| Kaiinite | 34 | 0.000 | 0.000 | 0.000 | 54 |

## What changed vs. the earlier capped-sample run (218,858 rows, 15k/class cap)

Overall accuracy dropped from 0.355 (capped) to 0.388 (full) -- actually
slightly *higher* at full scale, not lower. But per-class patterns shifted
in ways worth noting:

- **Dolomite got dramatically worse** at full scale: F1 0.055 (capped) ->
  0.009 (full). The capped run had *undersampled* Dolomite's 3,075 real
  samples down to whatever fit under the cap alongside majority classes,
  which apparently flattered it. At true scale, against the real class
  imbalance, Dolomite is essentially unlearned -- an even stronger
  Category-2/4 diagnosis candidate than first thought.
- **Coal improved** (F1 0.65 capped -> 0.394 full) -- wait, this direction
  reversed: capped run overstated Coal too, likely for the same
  undersampling-of-majority-classes reason (capping doesn't just shrink
  big classes, it changes the *relative* class balance the model sees
  during training, which changes what dampened weighting effectively does).
- **Bottom line: the capped run was not just "noisier," it was
  systematically distorted by the artificial per-class cap changing
  relative class weights.** Treat only this full-scale run as reportable;
  the earlier capped numbers should not be cited in the paper.

## Next steps
- Map all 22 classes into the four diagnostic categories formally.
- Test PEF inclusion (41% well coverage) specifically for Dolomite, given
  how much worse it looks now.
- Zero-recall classes (Conglomerate, Carnallite, Kaiinite) plus near-zero
  Gap and Dolomite are strong Category-3 (structural scarcity) candidates --
  though Dolomite has 3,075 samples, more than several classes that DO
  work (Anhydrite: 2,691, Coal: 5,113), so scarcity alone doesn't explain
  it -- lean toward Category 2 (missing feature) or 4 (ambiguity) pending
  the PEF test.
