# Geolink diagnostic category mapping (Phase 1, full-dataset baseline)
> Update (2026-08-31): the PEF-inclusion test (geolink_pef_test_2026-08-31.md) ruled out category 2 for Dolomite and Calcareous Cement, and both were reassigned to category 4. The rows and summary below still show the earlier, superseded assignment for those two classes.


Following the paper's established methodology (Section 2.3): every
underperforming class is diagnosed into one of the four categories *before*
any remedy is tested. A remedy is only applied where the diagnosis
indicates a tractable, testable intervention (Category 1 or 2). Categories
3 and 4 are reported transparently, not treated as targets for more model
complexity -- same discipline as the SEG and FORCE sections.

Based on: geolink/geolink_phase1_full_results_2026-08-30.md (full dataset,
1,184,461 rows, 181 wells, 22 classes).

| Lithology | F1 | Support | Diagnosis | Category |
|---|---|---|---|---|
| Anhydrite | 0.651 | 2,691 | Adequate baseline performance | -- |
| Limestone | 0.513 | 44,152 | Adequate baseline performance | -- |
| Shaly Silt | 0.473 | 276,702 | Adequate baseline performance (largest well-behaved class) | -- |
| Marlstone | 0.448 | 121,200 | Adjacency confusion with Argillaceous Limestone / Shale-family classes | 1 |
| Porous Limestone | 0.425 | 10,709 | Adjacency confusion with Limestone / Argillaceous Limestone | 1 |
| Silty Shale | 0.432 | 92,510 | Genuine grain-size continuum with Shaly Silt / Silt | 4 |
| Silt | 0.396 | 288,792 | Genuine grain-size continuum (largest class, one continuum endpoint) | 4 |
| Coal | 0.394 | 5,113 | Adequate baseline performance despite modest support | -- |
| Cross Bedded Sst | 0.351 | 75,559 | Adjacency confusion with Sandstone | 1 |
| Silty Sand | 0.330 | 83,252 | Genuine grain-size continuum with Sandstone / Sandy Silt | 4 |
| Sandstone | 0.303 | 36,550 | Adjacency confusion with Cross Bedded Sst / Silty Sand | 1 |
| Halite | 0.303 | 1,659 | Structural data scarcity (1,659 samples) | 3 |
| Argillaceous Limestone | 0.251 | 39,655 | Adjacency confusion with Limestone / Marlstone | 1 |
| Cinerite | 0.256 | 15,408 | Missing discriminating feature suspected -- volcanic ash typically needs mineralogy/spectral logs (THOR, URAN -- only 34/223 wells) not in baseline set | 2 |
| Chalk | 0.245 | 13,530 | Adjacency confusion with Limestone (both carbonates) | 1 |
| Calcareous Cement | 0.063 | 26,222 | Missing discriminating feature suspected -- diagenetic cementation likely needs a feature (e.g. PEF) not in baseline set, given decent support but very low F1 | 2 |
| Sandy Silt | 0.073 | 45,601 | Genuine ambiguity -- worst-performing transitional midpoint of the Sandstone-Silt continuum | 4 |
| Conglomerate | 0.000 | 1,013 | Structural data scarcity | 3 |
| Gap | 0.004 | 600 | Likely a data-quality/unclassified-interval code rather than a true lithology -- verify against the code table before treating as Category 3 | 3 (tentative) |
| Carnallite | 0.000 | 414 | Structural data scarcity (rare evaporite) | 3 |
| Kainite | 0.000 | 54 | Structural data scarcity (extreme -- fewer samples than any FORCE or SEG minority class) | 3 |
| **Dolomite** | **0.009** | **3,075** | **Flagged for targeted test, not yet assigned final category.** More support than Anhydrite (2,691) and Coal (5,113), both of which work -- scarcity alone does not explain the near-zero F1. Leading hypothesis: Category 2 (missing feature -- PEF, 41% well coverage). PEF-inclusion test needed before finalizing as Category 2 vs. Category 4. | 2 (pending test) |

## Category summary

- **Category 1 (adjacency confusion):** Marlstone, Porous Limestone, Cross Bedded Sst, Sandstone, Argillaceous Limestone, Chalk -- 6 classes, all within the carbonate family or the sandstone/cross-bedded pair
- **Category 2 (missing feature):** Cinerite, Calcareous Cement, and Dolomite pending confirmation -- all point toward PEF or mineralogy logs absent from the 5-curve baseline set
- **Category 3 (structural scarcity):** Halite, Conglomerate, Carnallite, Kainite, and tentatively Gap -- all under ~1,700 samples or confined to a handful of wells
- **Category 4 (genuine ambiguity):** Silty Shale, Silt, Silty Sand, Sandy Silt -- the siliciclastic grain-size continuum, directly analogous to FORCE's Sandstone/Shale finding
- **Adequate baseline, no diagnosis needed:** Anhydrite, Limestone, Shaly Silt, Coal

## Next step
Run the PEF-inclusion test on the subset of wells that have it (41% coverage,
per the curve survey), focused on Dolomite and Calcareous Cement -- the two
Category 2 candidates -- same targeted-remedy approach as FORCE's PEF test
for Chalk and Anhydrite.
