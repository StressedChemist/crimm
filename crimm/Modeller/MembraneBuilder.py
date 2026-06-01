"""Build protein-membrane systems using native crimm objects."""

from dataclasses import dataclass, field
import numpy as np

from crimm.StructEntities.Chain import Lipid
from crimm.StructEntities.Model import Model
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.Modeller.Solvator import Solvator


@dataclass
class MembraneSpec:
    """User-facing settings for building a membrane system."""

    upper_leaflet: dict[str, int]
    lower_leaflet: dict[str, int]

    box_xy: tuple[float, float]
    box_z: float

    lipid_z: float = 20.0
    protein_exclusion_radius: float = 2.8

    water_solvcut: float = 2.8
    salt_concentration: float = 0.15
    cation: str = "POT"
    anion: str = "CLA"


@dataclass
class MembraneBuildResult:
    """Summary of what was built."""

    model: Model
    lipid_count: int
    water_count: int | None = None
    ion_count: int | None = None


class MembraneBuilder:
    """Build a protein embedded in a lipid membrane."""

    def __init__(self, model: Model, spec: MembraneSpec):
        self.model = model
        self.spec = spec
        self.topology = TopologyGenerator()

    def build(self) -> MembraneBuildResult:
        """Run the full membrane-building workflow."""

        self.prepare_topology()
        self.add_lipid_bilayer()
        self.solvate()
        self.add_ions()

        return MembraneBuildResult(
            model=self.model,
            lipid_count=self.count_lipids(),
            water_count=self.count_waters(),
            ion_count=self.count_ions(),
        )

    def prepare_topology(self):
        """Load protein, lipid, water, and ion topology definitions."""

        # Later this should load protein topology if needed.
        # For first version, assume the input protein model already has topology.
        self.topology.load_residue_definitions("Lipid")
        self.topology.load_residue_definitions("Solvent")

    def add_lipid_bilayer(self):
        """Create upper and lower leaflet lipid residues."""

        lipid_chain = Lipid("MEMB")
        lipid_chain.pdbx_description = "generated membrane"

        resseq = 1

        for resname, count in self.spec.upper_leaflet.items():
            coords = self._leaflet_grid(count, z=self.spec.lipid_z)
            for coord in coords:
                residue = self._create_lipid_residue(resname, resseq, coord)
                lipid_chain.add(residue)
                resseq += 1

        for resname, count in self.spec.lower_leaflet.items():
            coords = self._leaflet_grid(count, z=-self.spec.lipid_z)
            for coord in coords:
                residue = self._create_lipid_residue(resname, resseq, coord)
                self._flip_lipid_z(residue)
                lipid_chain.add(residue)
                resseq += 1

        self.model.add(lipid_chain)
        self.topology.generate(lipid_chain)

    def solvate(self):
        """Add water around the protein-membrane system."""

        solvator = Solvator(self.model)
        solvator.solvate(
            box_type="tetra",
            box_dims=(
                self.spec.box_xy[0],
                self.spec.box_xy[1],
                self.spec.box_z,
            ),
            orient_coords=False,
            solvcut=self.spec.water_solvcut,
        )

    def add_ions(self):
        """Replace waters with ions."""

        solvator = Solvator(self.model)
        solvator.add_ions(
            concentration=self.spec.salt_concentration,
            cation=self.spec.cation,
            anion=self.spec.anion,
        )

    def _create_lipid_residue(self, resname: str, resseq: int, center):
        """Create one lipid residue and translate it to a target center."""

        lipid_defs, _ = self.topology.load_residue_definitions("Lipid")
        residue = lipid_defs[resname].create_residue(resseq=resseq, segid="MEMB")

        if residue is None:
            raise ValueError(f"Could not create lipid residue {resname}")

        self._translate_residue_to_center(residue, center)
        return residue

    def _leaflet_grid(self, count: int, z: float):
        """Return XY grid coordinates for one leaflet."""

        x_len, y_len = self.spec.box_xy
        n_side = int(np.ceil(np.sqrt(count)))

        xs = np.linspace(-x_len / 2, x_len / 2, n_side, endpoint=False)
        ys = np.linspace(-y_len / 2, y_len / 2, n_side, endpoint=False)

        coords = []
        for x in xs:
            for y in ys:
                coords.append(np.array([x, y, z], dtype=float))
                if len(coords) == count:
                    return coords

        return coords

    @staticmethod
    def _translate_residue_to_center(residue, target_center):
        atoms = list(residue.get_atoms())
        coords = np.array([atom.coord for atom in atoms])
        current_center = coords.mean(axis=0)
        shift = target_center - current_center

        for atom in atoms:
            atom.coord = atom.coord + shift

    @staticmethod
    def _flip_lipid_z(residue):
        atoms = list(residue.get_atoms())
        coords = np.array([atom.coord for atom in atoms])
        center_z = coords[:, 2].mean()

        for atom in atoms:
            atom.coord[2] = center_z - (atom.coord[2] - center_z)

    def count_lipids(self):
        return sum(
            len(chain)
            for chain in self.model
            if getattr(chain, "chain_type", None) == "Lipid"
        )

    def count_waters(self):
        return sum(
            len(chain)
            for chain in self.model
            if getattr(chain, "chain_type", None) == "Solvent"
        )

    def count_ions(self):
        return sum(
            len(chain)
            for chain in self.model
            if getattr(chain, "chain_type", None) == "Ion"
        )


def build_membrane_system(model: Model, spec: MembraneSpec) -> MembraneBuildResult:
    """Convenience function for one-call membrane building."""

    builder = MembraneBuilder(model, spec)
    return builder.build()
