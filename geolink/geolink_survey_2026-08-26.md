# Geolink dataset survey (223 wells, North Sea)

Pulled directly from the raw LAS archive via the public Google Drive
folder shared by GEOLINK (retrieved 2026-08-26). This is ground truth
from the actual files, not secondary blog/tutorial sources.

## Source and license

- 223 per-well LAS files (not a flattened CSV -- needs `lasio` to parse)
- **License is genuinely ambiguous in the primary materials themselves:**
  `License README.docx` states NLOD 2.0 and requires citing
  https://github.com/bolgebrygg/MalenoV ; `README_FIRST.txt` in the same
  zip states CC-BY-4.0. Confirm directly with GEOLINK before citing a
  license in the paper's Declarations section.
- Attribution required either way: https://www.geolink-s2.com/

## Curve availability (out of 223 wells)

| Curve | Wells present | Coverage |
|---|---|---|
| DEPT | 223 | 100.0% |
| LITHOLOGY_GEOLINK | 223 | 100.0% |
| GR | 205 | 91.9% |
| RDEP | 204 | 91.5% |
| RHOB | 200 | 89.7% |
| RMED | 200 | 89.7% |
| DTC | 198 | 88.8% |
| CALI | 196 | 87.9% |
| NPHI | 188 | 84.3% |
| DRHO | 187 | 83.9% |
| RSHA | 154 | 69.1% |
| SP | 147 | 65.9% |
| BS | 106 | 47.5% |
| PEF | 91 | 40.8% |
| DTE | 59 | 26.5% |
| ROP | 53 | 23.8% |
| DTS | 49 | 22.0% |
| DCAL | 35 | 15.7% |
| RMIC | 35 | 15.7% |
| THOR | 34 | 15.2% |
| URAN | 34 | 15.2% |
| SGR | 18 | 8.1% |
| RXO | 14 | 6.3% |
| MUDWEIGHT | 14 | 6.3% |
| DT | 7 | 3.1% |
| RHOM | 4 | 1.8% |
| CGR | 1 | 0.4% |
| TGAS | 1 | 0.4% |

PEF present in only 91/223 wells (41%) -- structurally similar to FORCE's
missingness pattern, so the Category-2 (missing discriminating feature)
diagnostic applies here too, and any PEF-based remedy only reaches a
minority of wells.

## Lithology class distribution

36 classes are defined in `Lithology code data.xlsx`, but only
**26 classes actually occur** in the data (1,650,950 total
samples) -- matching Carvalho et al.'s (ref [9]) count of 26, not the
36-class figure from secondary write-ups.

| Lithology | Code | Samples | % of dataset |
|---|---|---|---|
| Silt | 5.0 | 380,624 | 23.055% |
| Shaly Silt | 6.0 | 362,936 | 21.983% |
| Marlstone | 13.0 | 179,289 | 10.86% |
| Silty Shale | 7.0 | 114,576 | 6.94% |
| Silty Sand | 2.0 | 113,484 | 6.874% |
| Cross Bedded Sst | 3.0 | 89,793 | 5.439% |
| Limestone | 11.0 | 70,699 | 4.282% |
| Argillaceous Limestone | 12.0 | 70,647 | 4.279% |
| Sandy Silt | 4.0 | 61,089 | 3.7% |
| Sandstone | 1.0 | 47,890 | 2.901% |
| Cinerite | 19.0 | 38,727 | 2.346% |
| Calcareous Cement | 16.0 | 32,387 | 1.962% |
| Halite | 21.0 | 29,300 | 1.775% |
| Chalk | 9.0 | 25,582 | 1.55% |
| Porous Limestone | 10.0 | 14,992 | 0.908% |
| Coal | 18.0 | 5,964 | 0.361% |
| Anhydrite | 22.0 | 3,785 | 0.229% |
| Dolomite | 15.0 | 3,477 | 0.211% |
| Gap | 26.0 | 2,620 | 0.159% |
| Conglomerate | 17.0 | 1,927 | 0.117% |
| Carnallite | 33.0 | 865 | 0.052% |
| Polyhalite | 32.0 | 230 | 0.014% |
| Kaïnite | 34.0 | 54 | 0.003% |
| Volcanic Tuff | 20.0 | 10 | 0.001% |
| Metamorphic Rock | 30.0 | 2 | 0.0% |
| Plutonic Rock | 24.0 | 1 | 0.0% |

Silt + Shaly Silt alone make up 45% of all samples. The tail is extreme:
Plutonic Rock has exactly 1 sample total across all 223 wells -- a more
severe structural-scarcity case than FORCE's Basement class, and a clean
stress test for the paper's Category-3 diagnosis.

## Next step
Build a combined per-depth dataframe from the LAS files and run the
Phase 1 baseline + diagnostic mapping, same approach as SEG and FORCE.