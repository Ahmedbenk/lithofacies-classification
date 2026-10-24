# Geolink Phase 1 baseline -- FULL dataset (via Kaggle)

Run on Kaggle. Full
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
| Kainite | 34 | 0.000 | 0.000 | 0.000 | 54 |
