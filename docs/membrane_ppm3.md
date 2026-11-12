# Planar PPM3 and mixed leaflets

CRIMM uses PPM code obtained from Andrei Lomize and Irina Pogozheva, the main authors.
The native revision is `5784f088032e5b9fb35147815f67500b8cc94855`.
There is no Python scorer or fallback. PPM3 is distributed under GNU GPL
version 3, the same terms as PPM2, as confirmed by the CRIMM maintainer on
2026-11-12. See the [component notice](../crimm/Modeller/_ppm3/NOTICE.md)
and [license](../crimm/Modeller/_ppm3/LICENSE).

## Protein positioning

```python
from crimm.Modeller import MembraneOrienter

orientation = MembraneOrienter(model, membrane_type="OPC", topology="in").orient()
```

`OPC` is the DOPC code and the default. The case-sensitive planar codes are:

```text
PMm PMp PMf ERf ERm GOL LYS END VAC MOM MIM THp THb
GnO GnI GpI ARC LPC MPC OPC EPC
```

Blank codes and `MIC` are not supported. Curvature and multiple membranes are
outside this implementation. Each code selects equilibrium thickness and mismatch
stiffness. The beta-structure override remains active. Results record nominal and
effective thickness, stiffness, source revision, membrane code, and topology reference.

The transfer profile remains symmetric DOPC for every code. It does not accept
lipid percentages or separate leaflet profiles. The explicit CHARMM bilayer uses
independent mixtures. A membrane code is not a biological composition preset.
The separate [OPRLM preset catalog](membrane_presets.md) provides the published
explicit compositions and maps them to these native codes.

The topology reference is the first available backbone N of the first protein
subunit. Its original chain and residue identity appear in the result. For a
fragment, this is not necessarily the complete protein's biological N terminus.
`in` places it at negative Z. `out` places it at positive Z. Positive Z is the
outer leaflet. A reference within 0.002 angstroms of the midplane produces an
ambiguity error before coordinates change. No arbitrary-residue selector is included.

Resolve every histidine first. The hook accepts explicit HSE only and rejects
HSD, HSP, and unresolved HIS. It retains charges, caps, identities, and all
components. Caps and cofactors do not score. These are native model limits, not
biological protonation decisions. Source builds require gfortran. Local wheels
require its compatible runtime. See the [reference tests](../tests/test_native_membrane_orientation.py).

## Explicit leaflet mixtures

```python
from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec

spec = MembraneSpec(
    box_xy=(40, 40), box_z=80,
    upper_lipid_ratios={"POPC": 40, "DOPC": 40, "CHL1": 20},
    lower_lipid_ratios={"DMPC": 25, "DPPC": 25, "DSPC": 30, "CHL1": 20},
    lipid_z=20, membrane_core_z=15, random_seed=12345,
)
built = MembraneBuilder(model, spec).build()
print(built.leaflet_composition)
```

Supported packaged types are CHL1, DMPC, DOPC, DPPC, DSPC, and POPC.
Ratios are molecular proportions, including cholesterol. 80/20, 8/2, and 0.8/0.2
are equivalent. These examples are artificial validation mixtures, not biological presets.

Global `lipid_ratios` remains the shared default. Leaflet-specific ratios replace
it for that leaflet. Existing integer `upper_leaflet` and `lower_leaflet` counts
remain supported. An explicit count and a specific ratio for the same leaflet
conflict. An empty explicit leaflet stays empty. Empty or zero-sum ratios are invalid.

Counts use available leaflet area, weighted area per molecule, and largest-remainder
rounding. Ties retain input order. Records include requested percentages, counts,
realized percentages, and area per molecule. Head planes remain `+lipid_z` and
`-lipid_z`. They are not the native hydrophobic thickness.

The packaged NPZ retains all 2,000 conformers for each type, without water positions.
Solvator generates whole TIP3 waters. Every retained water and ion atom must stay
outside `abs(z) < membrane_core_z`. Parameter and conformer coverage is checked
before membrane residues enter the model.

## CHARMM preparation

```bash
python scripts/membrane_pycharmm.py --case helix --topology out \
  --ppm-membrane OPC \
  --upper-lipid-ratios '{"POPC":40,"DOPC":40,"CHL1":20}' \
  --lower-lipid-ratios '{"DMPC":25,"DPPC":25,"DSPC":30,"CHL1":20}' \
  --charmm-lib /path/to/charmm/lib
```

The environment needs pyCHARMM and NetworkX for the existing ring check. The
[example](../scripts/membrane_pycharmm.py) exports parameters and loads them in a
fresh CHARMM process. Shared atom types and parameter payloads load once. No
manual edits are required. The calculation uses BOMLEV 0 after the existing
water-ion stream restores its optional cross-force-field NBFIX warning levels.
The example uses 200 conformer retries. The builder default remains 20. Local
repacking uses 2 angstrom lateral moves. This resolves grid sites that the old
1 angstrom moves cannot clear. Counts, head-plane Z positions, random seeds,
and clash cutoffs are unchanged.

Checks cover all six types, eight CHL1 stereocentres, DOPC and POPC cis bonds,
head direction, exact leaflet composition, and periodic ring penetration. Lipid
P and sterol O3 restraints act along Z. Protein heavy-atom restraints act along
XYZ. Temporary signed CHL1 chirality guards prevent the inversions observed in
the unguarded six-lipid test. Their energies are checked against native CDIH.
Sterol double bonds do not receive PC cis guards.

The existing preparation schedule and GRMS target of 1.0 remain unchanged. Released
energies and gradients are separate. This is restrained preparation, not dynamics,
equilibration, or unrestrained convergence.

The acceptance matrix uses three seeds (12345, 12346, 12347) for six cases: pure
POPC control and helix, symmetric six-lipid control, asymmetric control, and
asymmetric helix with N-in and N-out. See [CHARMM testing](membrane_charmm_testing.md).
