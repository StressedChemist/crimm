# crimm

**crimm** stands for **Chemistry with the ReInvented Macromolecular Mechanics**.

This is a Python toolkit for biomolecule structure preparation, designed to unify common modeling routines under one scriptable platform. While many tools exist for tasks like solvation, adding hydrogens, or building missing loops, they often lack the scriptability needed for high-throughput pipelines. crimm aims to fill this gap by providing intuitive Python APIs that integrate seamlessly with pyCHARMM, RDKit, and other computational chemistry tools.

## Why crimm?

- **Scriptable**: All preparation steps are Python functions—no clicking through web interfaces or writing shell scripts
- **Integrated**: Works natively with pyCHARMM for energy calculations and simulations
- **Accurate**: Uses mmCIF format to correctly identify chain types, detect missing loops, and handle biological assemblies
- **Extensible**: Adaptors connect crimm to RDKit, PropKa, OpenMM, and more

## Features

- Fetch structures from RCSB PDB or AlphaFold Database
- Organize mmCIF structures into protein, nucleic acid, ligand, solvent, and ion components
- Generate **CHARMM36m** topology for proteins, canonical DNA/RNA, and protein-containing systems
- Normalize common terminal caps and report unsupported modified chemistry explicitly
- Parameterize small-molecule ligands with **CGenFF** integration, including lone-pair-aware workflows
- Read, write, and reload native CHARMM PSF and CRD files
- Solvate in cubic or truncated octahedral water boxes
- Add monovalent ions at target concentrations (SPLIT, SLTCAP methods)
- Build missing loops from homology models
- Visualize structures in Jupyter notebooks with NGLView

## Installation

```bash
pip install crimm
```

Optional cheminformatics and protonation extras:

```bash
pip install "crimm[all]"
```

Requires Python >= 3.9. For a complete development environment:

```bash
conda env create -f env.yaml
```

> **Note**: pyCHARMM, OpenMM, and the external CGenFF executable must be installed separately when those workflows are needed.

## Quick Example

```python
from crimm.Fetchers import fetch_rcsb
from crimm.Modeller import TopologyGenerator
from crimm.Modeller.Solvator import Solvator
from crimm.IO import write_psf, write_crd

# Fetch and prepare structure
model = fetch_rcsb('1LSA', organize=True)
topo_gen = TopologyGenerator()
for chain in model.protein:
    topo_gen.generate(chain)

# Solvate and add 150 mM KCl
solvator = Solvator(model)
solvator.solvate(cutoff=10.0, box_type='octa')
solvator.add_ions(concentration=0.15, cation='POT', anion='CLA')

# Write CHARMM files
write_psf(model, 'system.psf')
write_crd(model, 'system.crd')
```

## Membrane lipid data

The [OPRLM presets](docs/membrane_presets.md) cover 18 biological recipes and five
adjustable PC/cholesterol families. `MembraneSpec.from_preset("ERm", ...)` selects
the published mammalian ER composition. Biological presets use a separate compact
library with 105 complete lipid types and their reference parameters.

`MembraneBuilder` loads the packaged `lipid_conformers.npz` resource once per builder.
The file retains all 2,000 conformers for each of `CHL1`, `DMPC`, `DOPC`, `DPPC`, `DSPC`, and `POPC`.
It stores one atom-name list per lipid type and compressed coordinate arrays without precision loss.
It contains no water coordinates.
The existing Solvator generates the water positions.

`lipid_z` sets the head positions at `+lipid_z` and `-lipid_z` in Å.
The membrane core is a separate slab: `abs(z) < membrane_core_z`.
If any water atom enters this slab, the builder removes the complete water molecule by default.
The existing `remove_membrane_core_waters=False` setting disables this cleanup.

The original `.tar.gz` format remains supported through `lipid_library_path`.
The loader reads that archive once and caches the lipid coordinates.
To convert a legacy archive, run:

```bash
python -m scripts.pack_lipid_library /path/to/lipid_lib.tar.gz /path/to/lipid_conformers.npz
```

## Native membrane orientation

`MembraneOrienter` uses the PPM3 Fortran positioning model.
The Python energy approximation and its grid configuration no longer exist.
PPM3 uses a symmetric DOPC profile and its own ionization model, not CHARMM partial charges.
The default membrane code is `OPC`. The 21 named planar codes select native
matching thickness and stiffness. They do not select explicit lipid percentages.
The first available backbone N of the first protein subunit defines topology.
`in` gives negative Z and `out` gives positive Z. Positive Z is the outer leaflet.
The defaults are pH 7.5 and 293 K.
The temperature affects the native residue-ionization calculation.
The membrane profile retains its original empirical constants.

For a source checkout, build the native library:

```bash
python setup.py build_ext --inplace
```

Source builds require `gfortran`.
Compiled wheels contain the native library and residue data.
PPM3 is distributed under GNU GPL version 3, the same terms as PPM2, as
confirmed by the CRIMM maintainer. CRIMM uses PPM code obtained from Andrei
Lomize and Irina Pogozheva, the main authors. See the
[component notice](crimm/Modeller/_ppm3/NOTICE.md).
Binary installs require a compatible Fortran runtime.
The tested Linux build uses `libgfortran.so.5`.
An absent library produces an error without a Python fallback.

```python
from crimm.Modeller import MembraneOrienter

orientation = MembraneOrienter(model, ph=7.5, temperature=293.0,
                               membrane_type="OPC", topology="in").orient()
print(orientation.membrane_thickness)
```

The result contains `rotation`, `translation`, and the native positioning metrics.
The complete transform is `x_new = rotation @ x_old + translation`.
The `in_place=False` configuration returns the transform without a coordinate change.
The transform includes every retained component, cap atom, and hydrogen.
Only complete protein-core heavy atoms contribute to the native score.
Terminal cap atoms and retained cofactors do not contribute to this score.

The current hook supports neutral `HSE` through native `HIS` typing.
It rejects `HSD`, `HSP`, unresolved `HIS`, and unsupported side-chain patches.
The hook does not change residue identities or protonation states.
Resolve these states before the native call.
The result reports the optimizer tilt, not a fitted transmembrane-axis tilt.
The `positioning_uncertainty` value is Z-shift RMS for peripheral model 1 and thickness-range half-width for models 2 and 3.
Native positional and tilt spreads are not statistical confidence intervals.
The membrane thickness is the native hydrophobic thickness in Å.
It does not set the builder's separate `lipid_z` head planes.

## Solvated membrane example with pyCHARMM

The [step-by-step notebook](tutorials/8.PPM3MembraneWorkflow.ipynb) includes interactive NGLView structures.
It covers native orientation, leaflet composition, generated water, parameter export, and CHARMM minimization.

The [runnable example](scripts/membrane_pycharmm.py) builds, exports, loads, and minimizes one periodic membrane.
POPC remains the default. Separate `upper_lipid_ratios` and `lower_lipid_ratios`
support CHL1, DMPC, DOPC, DPPC, DSPC, and POPC mixtures. Ratios are molecular
proportions, including cholesterol. Results record requested and realized composition.
See [the mixture API and PPM3 limits](docs/membrane_ppm3.md).
It uses generated TIP3 water and 0.15 M KCl.
The exported files include the required topology and parameters.
No manual parameter-file correction is part of this workflow.

From the source checkout, run the capped transmembrane-helix case:

```bash
python scripts/membrane_pycharmm.py --case helix --charmm-lib /path/to/charmm/lib
```

The library directory must contain `libchmm.so` and its required runtime dependencies.
The Python environment must contain pyCHARMM.
The native PPM3 library must also exist for the helix case.
The Python environment also needs NetworkX for the existing ring check.
The example uses the offline 1gzm reference archive and requires no structure download.
It declares HSE65 and ACE/CT3 caps before atom construction.
This fragment is a test model, not a complete rhodopsin simulation model.

For the protein-free control, use `--case control`.
For a fixed seed, use `--seed 12345`.
For an explicit output location, use `--output /path/to/new_directory`.
The example refuses to overwrite an existing directory.
Use `--upper-lipid-ratios` and `--lower-lipid-ratios` for JSON mixtures.
Use `--topology in` or `--topology out` for the N reference direction.
The [matrix runner](scripts/membrane_matrix.py) runs all 18 wet preparation cases.

The output contains the PSF, initial CRD, parameter files, CHARMM log, restraint reference, minimized CRD, and `result.json`.
The example defines the crystal, periodic images, and PME before the energy calculation.
The initial packing test uses periodic heavy-atom contacts.
The result also records initial lipid hydrogen overlaps. Initial coordinates are not ready for dynamics.
Lipid phosphorus restraints act along Z only. Protein heavy-atom restraints act along X, Y, and Z.
Both selections retain one fixed reference while their force constants decrease through 10, 1, and 0.1.
Hydrogen relaxation runs first with heavy atoms fixed. The example then releases these fixed atoms and rebuilds the nonbonded lists.
Native dihedral restraints protect POPC cis bonds and glycerol chirality during preparation.
They reuse the local CHARMM-GUI reference, with `FORCE 250`, cis target 0°, and chirality target 120°.
A native MMFP soft guard restrains water oxygens outside `abs(z) = membrane_core_z + 2.0` during initialization.
Its force constant is 10 kcal mol⁻¹ Å⁻².
The example requires a final RMS gradient of at most 1.0 kcal mol⁻¹ Å⁻¹ with these restraints.
It then clears all fixed atoms and restraints and records the unrestrained energy and gradient without another minimization.
The final geometry test still requires every water atom outside the declared membrane core.
The final contact test includes every lipid atom, including hydrogen.
Other tests cover lipid chirality, cis geometry, ion counts, neutrality, and protein placement.
The result applies to the declared fixture, not to bilayer equilibration or the complete membrane acceptance matrix.
The [CHARMM testing guide](docs/membrane_charmm_testing.md) lists the checks and their limits.

## Modules

| Module | Purpose |
|--------|---------|
| `Fetchers` | Download structures from RCSB PDB or AlphaFold |
| `Modeller` | Topology generation, PSF/CRD loading, solvation, loop building |
| `IO` | Read/write PDB, mmCIF, PSF, and CRD files |
| `Adaptors` | Connect to pyCHARMM, RDKit, PropKa |

## Documentation

See `tutorials/` for Jupyter notebooks on structure preparation, topology generation, PSF/CRD workflows, loop building, and more.

## License

CRIMM: GPL-3.0-or-later. Bundled PPM3: GNU GPL version 3. See the
[PPM3 component notice](crimm/Modeller/_ppm3/NOTICE.md).

## Links

- Repository: https://github.com/BrooksResearchGroup-UM/crimm
- Issues: https://github.com/BrooksResearchGroup-UM/crimm/issues
