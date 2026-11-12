# Membrane testing with CHARMM

The membrane workflow remains under development. Further checks and code optimization remain necessary.
Use [notebook 8](../tutorials/8.PPM3MembraneWorkflow.ipynb) for the step-by-step workflow and structure views.

## Requirements

Source builds require gfortran. The CHARMM example requires pyCHARMM, NetworkX, and a compatible `libchmm.so`.
Optional trajectory checks also require MDAnalysis. The native tests use the pinned archives in `tests/data`.
See the [PPM3 notice](../crimm/Modeller/_ppm3/NOTICE.md) for source and distribution terms.

## Regression checks

Run the fast membrane checks from the repository root:

```bash
python -m pytest -q -m "not slow" \
  tests/test_lipid_library.py tests/test_membrane_*.py \
  tests/test_oprlm_presets.py tests/test_terminal_patch_control.py \
  tests/test_benchmark_crash_attribution.py tests/test_toppar_export.py
```

Run the native orientation checks separately:

```bash
python -m pytest -q tests/test_native_membrane_orientation.py \
  tests/test_native_membrane_parameters.py tests/test_native_membrane_tm.py
```

These tests compile PPM3 and compare native metrics and transformed coordinates against a separate reference build.
They also check unsupported histidine states, malformed input, repeated calls, and native error returns.

## CHARMM preparation

```bash
python scripts/membrane_pycharmm.py --case helix \
  --output /path/to/new/results --charmm-lib /path/to/charmm/lib
```

The default helix uses capped residues 35–65 from 1gzm, explicit HSE65, and a small POPC bilayer.
This fragment serves as a test model. It does not represent complete rhodopsin.

The command exports the PSF, CRD, topology, parameters, and load stream.
A fresh CHARMM process loads these files and checks atom identities, charges, and finite energies.
It defines periodic images and PME before restrained minimization.
The final checks cover contacts, ring penetration, stereochemistry, leaflet direction, water exclusion, and protein placement.
The result file records failures by the active stage.

The restrained GRMS target is 1.0 kcal mol⁻¹ Å⁻¹.
The runner then removes restraints and records the unrestrained energy and gradient.
That final record does not establish unrestrained convergence or membrane equilibration.

## Broader checks

```bash
python scripts/membrane_matrix.py --charmm-lib /path/to/charmm/lib
python scripts/membrane_matrix.py --oprlm --charmm-lib /path/to/charmm/lib
```

The first command runs 18 cases across protein orientation, leaflet mixtures, and three seeds.
The second runs all 23 presets at three seeds, for 69 solvated membrane controls.
Its PC cases use 20 mol% cholesterol and 333.15 K. These are test settings, not biological recipes.
See the [preset guide](membrane_presets.md) for the separate experimental aPM area expansion and optional NVT/NPT trials.

## Recorded results and limits

The default POPC notebook completed all 16 code cells through minimization and geometry checks.
Its 10,582-atom system reached a restrained GRMS of 0.772 kcal mol⁻¹ Å⁻¹.
The settings sweep passed for all 23 presets.

The full slow suite did not finish. The 69-case CHARMM matrix and interactive browser checks remain incomplete.
The aPM preset remains experimental at reference density.
Box checks reject invalid dimensions and severe clashes.
Dedicated checks for protein-image clearance, water depth, and cutoff compatibility remain absent.
These results do not establish equilibrium or production readiness for every preset.
