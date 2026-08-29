# Geolink Phase 1 baseline results

Calibrated single XGBoost with dampened (sqrt) class weighting,
StratifiedGroupKFold (3 folds, grouped by well), same methodology as
SEG and FORCE. Feature set: GR, RDEP (log10), RHOB, NPHI, DTC -- the
5 curves with >=84% well coverage (see geolink_survey_2026-08-26.md).

**Note on scale:** run on a per-class-capped sample (15,000 rows/class,
218,858 rows total, 181 wells) rather than the full 1.18M-row cleaned
set, due to memory limits in the sandbox used to run this. Numbers below
are a first-pass diagnostic signal, not a final reportable result --
re-run on the full dataset (or with a larger-memory environment) before
this goes in the paper.

```
After capping: (218858, 8) | classes: 22 | wells: 181
Fold 1/3 done
Fold 2/3 done
Fold 3/3 done

======================================================================
GEOLINK PHASE 1 BASELINE (per-class capped at 15000) -- accuracy: 0.3546
======================================================================
             lithology  code  precision   recall       f1  support
             Sandstone   1.0   0.403012 0.426333 0.414345    15000
            Silty Sand   2.0   0.289555 0.292933 0.291234    15000
      Cross Bedded Sst   3.0   0.273415 0.329867 0.299000    15000
            Sandy Silt   4.0   0.182159 0.116533 0.142137    15000
                  Silt   5.0   0.240358 0.139600 0.176619    15000
            Shaly Silt   6.0   0.288649 0.316667 0.302009    15000
           Silty Shale   7.0   0.429195 0.543733 0.479722    15000
             Limestone  11.0   0.402952 0.475000 0.436020    15000
             Marlstone  13.0   0.322669 0.449800 0.375773    15000
Argillaceous Limestone  12.0   0.333607 0.271267 0.299224    15000
     Calcareous Cement  16.0   0.169801 0.082467 0.111016    15000
              Cinerite  19.0   0.511085 0.694667 0.588900    15000
                 Chalk   9.0   0.387848 0.278344 0.324096    13530
      Porous Limestone  10.0   0.446068 0.467644 0.456601    10709
                  Coal  18.0   0.570986 0.757481 0.651143     5113
              Dolomite  15.0   0.067637 0.046829 0.055342     3075
             Anhydrite  22.0   0.740804 0.800818 0.769643     2691
                Halite  21.0   0.637486 0.342375 0.445490     1659
          Conglomerate  17.0   0.002530 0.006910 0.003704     1013
                   Gap  26.0   0.003158 0.005000 0.003871      600
            Carnallite  33.0   0.000000 0.000000 0.000000      414
               Kaïnite  34.0   0.000000 0.000000 0.000000       54
DONE
```

## Early diagnostic read

- **Category 4 (genuine ambiguity) candidates:** Sandy Silt, Silt, Shaly
  Silt, Silty Sand, Cross Bedded Sst -- all cluster around F1 0.14-0.30,
  consistent with a siliciclastic grain-size continuum rather than sharp
  class boundaries (same pattern as FORCE's Sandstone/Shale).
- **Category 3 (structural scarcity) candidates:** Kaiinite (54 samples,
  F1=0), Carnallite (414, F1=0), Conglomerate (1,013, F1=0.004), Gap (600,
  F1=0.004) -- all near-zero, consistent with too little data to learn from
  regardless of method.
- **Dolomite (F1=0.055)** -- notably weak despite reasonable support (3,075
  samples), mirroring FORCE's Dolomite problem exactly. Worth checking
  whether this is Category 2 (missing feature -- no PEF in the baseline
  set here, and PEF only covers 41% of wells) before assuming Category 4.
- **Working baseline classes:** Anhydrite (F1=0.77), Coal (F1=0.65),
  Cinerite (F1=0.59) -- comparable to FORCE's better-behaved classes.

## Next steps
- Re-run on full 1.18M-row dataset once compute allows, to confirm these
  numbers hold at scale.
- Test PEF inclusion on the subset of wells that have it (41% coverage)
  to check whether Dolomite is Category 2 (missing feature) like it was
  hypothesized to be, but wasn't fully confirmed, on FORCE.
- Map every class above into the four diagnostic categories formally,
  same table format as the SEG/FORCE sections.