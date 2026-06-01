"""Build protein-membrane systems using native crimm objects."""

from dataclasses import dataclass

import numpy as np

from crimm.Modeller.Solvator import Solvator
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.StructEntities.Chain import Lipid, Sterol
from crimm.StructEntities.Model import Model


STEROL_RESNAMES = frozenset({"CHL1"})


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
    sterol_count: int
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
            sterol_count=self.count_sterols(),
            water_count=self.count_waters(),
            ion_count=self.count_ions(),
        )

    def prepare_topology(self):
        """Load lipid, sterol, water, and ion topology definitions."""

        self.topology.load_residue_definitions("Lipid")
        if self._has_sterols():
            self.topology.load_residue_definitions("Sterol")
        self.topology.load_residue_definitions("Solvent")

    def add_lipid_bilayer(self):
        """Create upper and lower leaflet lipid and sterol residues."""

        lipid_chain = Lipid("MEMB")
        lipid_chain.pdbx_description = "generated membrane phospholipids"

        sterol_chain = Sterol("CHOL")
        sterol_chain.pdbx_description = "generated membrane sterols"

        residue_numbers = {"Lipid": 1, "Sterol": 1}

        for resname, count in self.spec.upper_leaflet.items():
            coords = self._leaflet_grid(count, z=self.spec.lipid_z)
            for coord in coords:
                self._add_membrane_residue(
                    resname, coord, lipid_chain, sterol_chain, residue_numbers
                )

        for resname, count in self.spec.lower_leaflet.items():
            coords = self._leaflet_grid(count, z=-self.spec.lipid_z)
            for coord in coords:
                residue = self._add_membrane_residue(
                    resname, coord, lipid_chain, sterol_chain, residue_numbers
                )
                self._flip_lipid_z(residue)

        if len(lipid_chain) > 0:
            self.model.add(lipid_chain)
            self.topology.generate(lipid_chain)

        if len(sterol_chain) > 0:
            self.model.add(sterol_chain)
            self.topology.generate(sterol_chain)

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

    def _add_membrane_residue(
        self,
        resname: str,
        center: np.ndarray,
        lipid_chain: Lipid,
        sterol_chain: Sterol,
        residue_numbers: dict[str, int],
    ):
        chain_type = self._chain_type_for_resname(resname)
        chain = sterol_chain if chain_type == "Sterol" else lipid_chain
        segid = "CHOL" if chain_type == "Sterol" else "MEMB"
        resseq = residue_numbers[chain_type]

        residue = self._create_residue(resname, chain_type, resseq, segid, center)
        chain.add(residue)
        residue_numbers[chain_type] += 1
        return residue

    def _create_residue(
        self,
        resname: str,
        chain_type: str,
        resseq: int,
        segid: str,
        center: np.ndarray,
    ):
        """Create one membrane residue and translate it to a target center."""

        residue_defs, _ = self.topology.load_residue_definitions(chain_type)
        if resname not in residue_defs:
            raise ValueError(
                f"Residue {resname} is not available in {chain_type} topology."
            )

        residue = residue_defs[resname].create_residue(resseq=resseq, segid=segid)
        if residue is None:
            raise ValueError(f"Could not create membrane residue {resname}")

        self._translate_residue_to_center(residue, center)
        return residue

    def _leaflet_grid(self, count: int, z: float):
    """Return XY grid coordinates for one leaflet, avoiding protein XY footprint."""

    x_len, y_len = self.spec.box_xy
    n_side = int(np.ceil(np.sqrt(count * 2)))

    protein_xy = self._protein_xy_coords()

    xs = np.linspace(-x_len / 2, x_len / 2, n_side, endpoint=False)
    ys = np.linspace(-y_len / 2, y_len / 2, n_side, endpoint=False)

    coords = []
    for x in xs:
        for y in ys:
            point_xy = np.array([x, y], dtype=float)

            if self._too_close_to_protein(point_xy, protein_xy):
                continue

            coords.append(np.array([x, y, z], dtype=float))
            if len(coords) == count:
                return coords

    raise ValueError(
        f"Could only place {len(coords)} of {count} membrane residues. "
        "Increase box_xy or reduce protein_exclusion_radius."
    )

    def _has_sterols(self):
        resnames = set(self.spec.upper_leaflet) | set(self.spec.lower_leaflet)
        return any(self._is_sterol(resname) for resname in resnames)

    @staticmethod
    def _chain_type_for_resname(resname: str):
        return "Sterol" if MembraneBuilder._is_sterol(resname) else "Lipid"

    @staticmethod
    def _is_sterol(resname: str):
        return resname.upper() in STEROL_RESNAMES

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

    def count_sterols(self):
        return sum(
            len(chain)
            for chain in self.model
            if getattr(chain, "chain_type", None) == "Sterol"
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

