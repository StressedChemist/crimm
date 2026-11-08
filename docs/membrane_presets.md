# OPRLM membrane presets

CRIMM includes recipes for all systems from the [2025 OPRLM paper](https://doi.org/10.1016/j.jmb.2025.168966):
18 biological recipes and five adjustable PC/cholesterol families. These are 23 preset IDs, not 20 fixed recipes.

The biological recipes come from the [CHARMM-GUI biomembrane archive](https://www.charmm-gui.org/?doc=archive&lib=biomembrane).
The archive accompanies the [2022 reference study](https://doi.org/10.1021/acs.jcim.1c01514).
The packaged catalog records the source URLs, archive SHA256 hashes, temperatures, areas, and original table counts.
No runtime download is necessary.

The aPM recipe remains experimental at reference density. Strict packing has not passed at this density.
Its archived initial structure also contains sub-1.2 Å intermolecular heavy-atom contacts.
The factory reports this limit before construction. It does not silently lower counts or clash cutoffs.
The remaining named recipes still require a separate equilibration protocol before production dynamics.

## Select a preset

| ID | Biological membrane | Native PPM3 code |
| --- | --- | --- |
| PMm | Mammalian plasma | PMm |
| PMp | Plant/algal plasma | PMp |
| PMf | Fungal plasma | PMf |
| ERm | Mammalian endoplasmic reticulum | ERm |
| ERf | Fungal endoplasmic reticulum | ERf |
| GOLm | Mammalian Golgi | GOL |
| GOLf | Fungal Golgi | GOL |
| ENDm | Mammalian endosomal | END |
| LYSm | Mammalian lysosomal | LYS |
| VAC | Plant vacuole | VAC |
| MOM | Mitochondrial outer | MOM |
| MIM | Mitochondrial inner | MIM |
| THYp | Plant thylakoid | THp |
| THYb | Eubacterial thylakoid | THb |
| aPM | Archaeal plasma | ARC |
| G-OM | Gram-negative bacterial outer | GnO |
| G-IM | Gram-negative bacterial inner | GnI |
| G+PM | Gram-positive bacterial plasma | GpI |

The five PC families are `POPC`, `DOPC`, `DSPC`, `DMPC`, and `DPPC`.
Each accepts a cholesterol molecular percentage in `[0, 100)`.
Biological recipes reject cholesterol overrides. Custom recipes still use `MembraneSpec` directly.

```python
from crimm.Modeller import MembraneOrienter
from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.Modeller.MembranePresets import get_membrane_preset, list_membrane_presets

print(list_membrane_presets())
preset = get_membrane_preset("ERm")
orientation = MembraneOrienter(
    model, membrane_type=preset["ppm_membrane"],
    temperature=preset["temperature_K"], topology="in",
).orient()
cell = preset["reference_box_A"]
spec = MembraneSpec.from_preset(
    "ERm", box_xy=cell[:2], box_z=cell[2],
    random_seed=12345, max_repack_attempts=200,
)
builder = MembraneBuilder(model, spec)
result = builder.build()
print(result.leaflet_composition)
```

Resolve histidine states and build the complete protein topology before this example.
The native hook accepts explicit HSE only. It does not assign biological protonation states.

For a PC/cholesterol membrane, use an explicit percentage:

```python
preset = get_membrane_preset("POPC", cholesterol_percent=20)
spec = MembraneSpec.from_preset(
    "POPC", cholesterol_percent=20, box_xy=(80, 80), box_z=100,
    random_seed=12345, max_repack_attempts=200,
)
```

PC families have no reference temperature. Select a temperature that matches the intended physical phase.
DMPC uses native code `MPC`. DOPC uses `OPC`.
POPC, DPPC, and DSPC use `OPC` as an explicit positioning approximation.

## Keep orientation and composition separate

Positive Z is the outer, extracellular, or luminal leaflet. Negative Z is the inner or cytosolic leaflet.
`topology="in"` places the first available protein backbone N at negative Z. `"out"` places it at positive Z.
For a protein fragment, this reference is not necessarily the biological N terminus.

The preset preserves separate leaflet ratios and areas. It selects the matching native PPM3 code.
The native transfer profile remains symmetric DOPC for every code.
It does not describe the explicit asymmetric lipid composition. See the [native-model limits](membrane_ppm3.md).

Counts use available area and largest-remainder rounding. Small boxes can omit minor lipid components.
The published lipid areas do not sum to the area of each initial reference box.
The factory scales these areas uniformly within each leaflet to reproduce its initial reference inventory.
It retains the original published areas in the catalog. This initial density is not a claim about equilibrium membrane area.
The reference boxes provide a useful initial size, not a guarantee of exact reference counts around a protein.
Check `leaflet_composition` after construction.
Reference lipids use lateral retry moves scaled to their mean head spacing.
The builder places large molecules first. It can retry a head position elsewhere in the same periodic leaflet.
The clash cutoffs, lipid counts, and head-plane Z positions remain unchanged.

`lipid_z` sets the head planes. `membrane_core_z` sets a separate water-exclusion slab.
The aPM preset retains the source-relative head depths for archaeol, quinone, and archaeal phospholipids through `lipid_z_offsets`.
Changing `lipid_z` shifts all these planes by the same distance.
These remain explicit geometry settings, not automatic copies of the native hydrophobic thickness.
The current builder uses planar leaflets. It does not construct curved membranes or membrane-spanning tetraether models.

## Lipid data and force-field export

The biological library contains 105 complete reference lipid types.
It retains all available reference conformers, without water coordinates or repeated text structures.
The original six-type library remains unchanged, with 2,000 conformers per type.

The compiler groups covalently connected molecules before it removes solvent.
Glycolipid sugars remain attached to their lipid. LPS remains one complete molecule.
Compound molecules use one CHARMM residue with unique atom names.
Each template retains a per-atom source identity map, charges, masses, atom types, and bonded terms.
The compiler preserves stereochemistry through proper rotations, not coordinate reflections.

Some raw reference conformers contain mixed stereo states or incorrect double-bond configurations.
The builder selects only conformers that pass the recorded stereo checks. It retains every raw conformer in the package.
Consistent reference configurations define chirality. Source IC records resolve variable stereocentres and define double-bond targets.
This preserves the reference model conventions. It does not independently establish absolute stereochemical assignments.
LLPA, LLPE, LLPS, and PLPI have no compliant archived conformer.
The compiler adds one seed from each source IC table through the existing CRIMM coordinate builder.
The template marks these added seeds with `generated_conformer="source_IC"`.

Each biological preset loads its own reference parameter payload through the existing topology and parameter loaders.
The payload includes lipid extensions, carbohydrate parameters, CGenFF parameters, and the required shared atom types.
Parameters follow the loading order in each archive's `toppar.str`, not the order of files in the tar archive.
It does not combine different archive versions into one parameter set.
The compiler excludes protein-specific and nucleic-acid-specific terms from their reference parameter files.
It retains shared terms needed by the lipid atom types. The protein retains its existing backbone and CMAP parameters.
Check the exported shared parameters before use with a separately calibrated force field.
Shared protein nonbonded terms load once. Incompatible shared values cause an error before export.

Solvator generates all water positions. Immediately after construction, every retained water and ion atom lies outside the membrane core.
Presets retain the builder defaults for other solvation settings, including 0.15 M KCl.
G-OM requires five CAL ions per outer-leaflet LPS molecule, as in its reference system.
The builder first replaces eligible positive-Z waters with these calcium ions. It then adds monovalent salt and counterions.
This reproduces the calcium inventory, not the reference calcium binding sites.

Two source-label details require care:

- THYb table rows sum to 105 molecules, although its total is 100. Its PSF/CRD contains 40 MGDG-PO, not 45.
  The preset uses the PSF/CRD inventory and retains both values in `source_discrepancies`.
- PMm labels CMH as GlcCer. Its reference PSF contains CER160 linked to BGAL.
  The compiled molecule is `GALCER`. The catalog retains the original display label and its atom identity map.

## Run CHARMM preparation

```bash
python scripts/membrane_pycharmm.py --case helix --membrane ERm \
  --topology in --charmm-lib /path/to/charmm/lib

python scripts/membrane_pycharmm.py --case control --membrane POPC \
  --cholesterol-percent 20 --temperature 310.15 --charmm-lib /path/to/charmm/lib
```

The command exports the PSF, coordinates, topology, parameters, and loading stream.
It loads them in a fresh CHARMM process, checks their identities, and runs restrained minimization.
Biological presets use their reference box unless `--cell X Y Z` overrides it.
Named presets and NVT/NPT trials have a one-hour CHARMM time limit.
The earlier custom-mixture minimization command retains its 15-minute limit.
The command rejects a named preset combined with custom leaflet ratios.

An explicit preparation option can expand the initial membrane area without adding lipids:

```bash
python scripts/membrane_pycharmm.py --case control --membrane aPM \
  --initial-area-scale 1.05 --short-range-warmup --nvt-steps 100 --npt-steps 100 \
  --output /path/to/new/apm-trial --charmm-lib /path/to/charmm/lib
```

`--initial-area-scale 1.05` adds 5% area, not 5% to each box edge.
The builder resolves nominal lipid counts first. It then multiplies X and Y by the square root of this factor.
Z and intramolecular coordinates do not scale. Solvator generates water for the expanded box.
The Python equivalent is `MembraneSpec.from_preset(..., initial_area_scale=1.05)`.
`builder.spec` contains the expanded dimensions. The original spec remains unchanged.
The result records both the nominal cell and the expanded cell. The default factor remains 1.

`--short-range-warmup` uses a 6 Å cutoff without PME during hydrogen relaxation, with all heavy atoms fixed.
This optional local relaxation precedes full PME minimization. It does not change force-field parameters or packing acceptance cutoffs.
The runner restores the full PME settings before it releases the heavy atoms.
The final minimization and all dynamics stages use the full settings.
The option defaults to off. The [archived aPM inputs](https://www.charmm-gui.org/archive/biomembrane/aPM/aPM.tar.gz)
also use a short-range initial minimization, although their atom restraints and step count differ.

The optional dynamics steps use a 1-fs timestep. The example runs 0.1 ps of NVT and 0.1 ps of NPT.
This duration checks native integration only. The longer 1-ps/5-ps aPM pilot remains unvalidated.
NVT retains weak head-Z restraints, water-core restraints, and stereochemistry restraints.
NPT removes head-Z and water-core restraints. It retains weaker stereochemistry restraints.
NPT uses a tetragonal cell, with coupled X/Y dimensions and an independent Z dimension, at 1 atm.
It requires equal X and Y dimensions. MDAnalysis checks the trajectory coordinates and cell dimensions.
Each stage writes a restart, trajectory, coordinates, and the dynamics input.
Final geometry checks use the final coordinates and final cell, not the initial cell.
Both dynamics step counts default to zero. No automatic expansion or dynamics fallback occurs.
These short trials do not establish equilibrium. The aPM preset retains its experimental status.

Notebook [8.PPM3MembraneWorkflow.ipynb](../tutorials/8.PPM3MembraneWorkflow.ipynb) provides the same workflow with structure views.
Step 2 selects the preset. Its default remains a small POPC test.

The broader acceptance matrix is available separately:

```bash
python scripts/membrane_matrix.py --oprlm --charmm-lib /path/to/charmm/lib
```

This matrix runs all 23 presets at three seeds, for 69 solvated membrane controls.
Its PC cases use 20 mol% cholesterol and 333.15 K. These are explicit test settings, not paper-derived recipes.
This matrix does not test every cholesterol percentage or protein topology.
The command without `--oprlm` retains the earlier 18-case protein and mixture matrix.

Restrained minimization is not membrane equilibration or proof of production readiness.
The result record reports failures by the stage that actually ran.

The [CHARMM testing guide](membrane_charmm_testing.md) lists reproducible commands and current validation limits.

## Rebuild the compact data

```bash
python scripts/import_oprlm_reference.py --cache /path/to/new/cache \
  --output /path/to/new/cache/recipes.json --download-archives --audit
python scripts/compile_oprlm_library.py --cache /path/to/new/cache \
  --catalogue /path/to/new/cache/recipes.json --output crimm/Data/oprlm_lipids.npz
```

The importer records source hashes. The compiler checks archive hashes and molecule inventories before it writes package data.
