# Cost-based operating-point selection for water electrolysis

Reproducible implementation of a constraint-based program that selects the current
density of a water electrolysis cell by minimizing the levelized cost of hydrogen (LCOH).

The cost enters through one group, the ratio Γ of areal capital cost per operating hour to
electricity price (W cm⁻²). The objective is the cost-equivalent voltage

    U(j) = [V(j) + Γ/j] / η_F(j),        LCOH = 26.59 · c_el · U(j)

minimized over the current density j, subject to limits that belong to the cell architecture.

## What is included

| Folder | Content |
|---|---|
| `code/model.py` | cell voltage models, calibration, the operating-point program |
| `code/analysis.py` | all analyses and figures; writes `output/results.json` |
| `data/` | polarization data transcribed from open datasets (CC BY 4.0) |
| `output/` | numerical results and three summary tables (CSV) |
| `figures/` | calibration, objective, optimum versus Γ, coverage regimes |

## Archetypes

- Zero-gap alkaline cell, 80 °C (calibrated)
- PEM cell, 80 °C (calibrated; 60 and 70 °C curves used for a transfer test)
- AEM cell, 50 and 80 °C (calibrated)
- Finite-gap alkaline cell (derived from the zero-gap fit with an assumed gap resistance
  and voidage coefficient; not calibrated). Only this archetype carries a bubble-coverage
  constraint, θ = 0.023·i^0.3 with i in A m⁻² (Vogt and Balzer, 2005).

## Analyses

Held-out validation, extrapolation test, temperature and day-to-day transfer tests, residual
bootstrap (1000 resamples), sweep of Γ from 0.01 to 3 W cm⁻², coverage-ceiling sweep with the
cost of the ceiling, efficiency-floor thresholds, crossover and lifetime scenarios, and a joint
perturbation of parameters and cost ratio.

## Reproduction

    pip install -r requirements.txt
    cd code && python analysis.py

The random seed is fixed (42). The run takes about one minute and regenerates
`output/results.json` and the figures.

## Data sources

- Shinde, D. V., Smith, G. (2022). Long-term testing of alkaline electrolyser single cell using
  Raney Ni electrodes. Zenodo. https://doi.org/10.5281/zenodo.7074952
- Alanazi, M., Hughes, K. J., Pourkashanian, M. (2026). Verification, validation, and
  uncertainty-quantification dataset for a one-dimensional PEM electrolysis model. Zenodo.
  https://doi.org/10.5281/zenodo.22112171 (digitised points of García-Valverde et al., 2012,
  https://doi.org/10.1016/j.ijhydene.2011.09.027)
- Ranade, A., Tsampas, M. (2025). Nanostructured Ni-based alloys as electroactive porous
  transport layers for anion exchange membrane water electrolysis. Zenodo.
  https://zenodo.org/records/15125443

The data files keep the licence of their sources (CC BY 4.0). The code is released under the MIT licence.

## Limitations

Each calibrated archetype is one laboratory single cell at beginning of life. The cell model
has no mass-transport term, so optima above the calibrated current range are extrapolations
and are flagged in the outputs. The results do not rank the technologies.
