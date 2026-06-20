from __future__ import annotations

"""Build protein-membrane systems using native crimm objects."""

from dataclasses import dataclass
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial import KDTree

from crimm.Modeller.Solvator import Solvator
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.StructEntities.Chain import Lipid, Sterol
from crimm.StructEntities.Model import Model


STEROL_RESNAMES = frozenset({"CHL1"})

DEFAULT_LIPID_AREAS = {
    "POPC": 64.0,
    "POPE": 58.0,
    "POPG": 62.0,
    "POPS": 62.0,
    "DLPC": 63.0,
    "DLPE": 56.0,
    "DMPC": 60.0,
    "DOPC": 67.0,
    "DPPC": 64.0,
    "CHL1": 40.0,
}

@dataclass
class MembraneSpec:
    """User-facing settings for building a membrane system."""

    box_xy: tuple[float, float]
    box_z: float
    solvation_box: tuple[float, float, float] | None = None
    upper_leaflet: dict[str, int] | None = None
    lower_leaflet: dict[str, int] | None = None
    lipid_ratios: dict[str, float] | None = None
    area_per_lipid: float | None = None
    lipid_area_overrides: dict[str, float] | None = None
    account_for_protein_footprint: bool = True
    protein_footprint_sample_spacing: float = 2.0
    lipid_z: float = 20
    lipid_z_scale: float = 0.6
    random_seed: int = 12345
    use_lipid_library: bool = True
    lipid_library_path: str | None = None
    use_charmm_gui_head_positions: bool = False
    charmm_gui_head_crd: str | None = None
    align_head_positions_to_protein_xy: bool = True
    protein_exclusion_radius: float = 2.8
    clash_check: bool = True
    protein_clash_cutoff: float = 1.7
    membrane_clash_cutoff: float = 1.2
    max_repack_attempts: int = 20
    water_solvcut: float = 2.8
    remove_membrane_core_waters: bool = False
    membrane_core_z: float = 15.0
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


class LipidConformerLibrary:
    """Read CHARMM-GUI lipid conformers from lipid_lib.tar.gz."""

    def __init__(self, archive_path):
        self.archive_path = Path(archive_path)
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Lipid library not found: {self.archive_path}")

        self._members_by_resname = {}
        with tarfile.open(self.archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".crd"):
                    continue

                parts = member.name.split("/")
                if len(parts) < 4:
                    continue

                resname = parts[1].upper()
                self._members_by_resname.setdefault(resname, []).append(member.name)

    def random_conformer(self, resname: str, rng):
        resname = resname.upper()
        if resname not in self._members_by_resname:
            raise ValueError(f"No lipid conformers found for {resname}")

        member_name = rng.choice(self._members_by_resname[resname])
        return self._read_crd_member(member_name)

    def _read_crd_member(self, member_name):
        with tarfile.open(self.archive_path, "r:gz") as archive:
            handle = archive.extractfile(member_name)
            if handle is None:
                raise ValueError(f"Could not read {member_name}")

            lines = handle.read().decode("utf-8").splitlines()

        data_started = False
        coords = {}

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("*"):
                continue

            fields = stripped.split()

            if not data_started:
                data_started = True
                continue

            if len(fields) < 10:
                continue

            atom_name = fields[3].strip()
            coords[atom_name] = np.array(
                [float(fields[4]), float(fields[5]), float(fields[6])],
                dtype=float,
            )

        return coords

@dataclass
class PackedHeadPosition:
    resname: str
    coord: np.ndarray
    leaflet_sign: int

class MembraneBuilder:
    """Build a protein embedded in a lipid membrane."""

    def __init__(self, model: Model, spec: MembraneSpec):
        self.model = model
        self.spec = spec
        self.topology = TopologyGenerator()
        self.rng = np.random.default_rng(spec.random_seed)
        
        self.upper_leaflet = self._resolve_leaflet_counts("upper")
        self.lower_leaflet = self._resolve_leaflet_counts("lower")
    
        self.lipid_library = None
        if spec.use_lipid_library:
            lipid_library_path = spec.lipid_library_path
            if lipid_library_path is None:
                lipid_library_path = (
                    Path(__file__).resolve().parents[1]
                    / "charmm_gui_9qsy"
                    / "lipid_lib.tar.gz"
                )
            self.lipid_library = LipidConformerLibrary(lipid_library_path)

        self.packed_head_positions = None
        if spec.use_charmm_gui_head_positions:
            head_crd = spec.charmm_gui_head_crd
            if head_crd is None:
                head_crd = (
                    Path(__file__).resolve().parents[1]
                    / "charmm_gui_9qsy"
                    / "step3_packing_head.crd"
                )
            self.packed_head_positions = self._load_charmm_gui_head_positions(head_crd)

    def _resolve_leaflet_counts(self, leaflet: str):
        """Return explicit or estimated lipid counts for one leaflet."""

        if leaflet == "upper":
            explicit = self.spec.upper_leaflet
        elif leaflet == "lower":
            explicit = self.spec.lower_leaflet
        else:
            raise ValueError(f"Unknown leaflet: {leaflet}")

        if explicit is not None:
            return {resname.upper(): int(count) for resname, count in explicit.items()}

        if self.spec.lipid_ratios is None:
            raise ValueError(
                "Either upper_leaflet/lower_leaflet counts or lipid_ratios must be provided."
            )

        ratios = self._normalized_lipid_ratios()
        target_area = self._target_area_per_lipid(ratios)
        leaflet_area = self._available_leaflet_area()
        total_count = int(round(leaflet_area / target_area))

        if total_count <= 0:
            raise ValueError("Estimated leaflet lipid count must be positive.")

        ratios = {
            resname.upper(): float(ratio)
            for resname, ratio in self.spec.lipid_ratios.items()
        }

        ratio_sum = sum(ratios.values())
        if ratio_sum <= 0:
            raise ValueError("lipid_ratios must sum to a positive value.")

        normalized = {
            resname: ratio / ratio_sum
            for resname, ratio in ratios.items()
        }

        counts = {
            resname: int(np.floor(total_count * ratio))
            for resname, ratio in normalized.items()
        }

        remainder = total_count - sum(counts.values())
        if remainder > 0:
            order = sorted(
                normalized,
                key=lambda resname: (total_count * normalized[resname]) % 1,
                reverse=True,
            )
            for resname in order[:remainder]:
                counts[resname] += 1

        return {resname: count for resname, count in counts.items() if count > 0}

    def _normalized_lipid_ratios(self):
        """Return normalized lipid/sterol ratios from the membrane spec."""

        ratios = {
            resname.upper(): float(ratio)
            for resname, ratio in self.spec.lipid_ratios.items()
        }

        ratio_sum = sum(ratios.values())
        if ratio_sum <= 0:
            raise ValueError("lipid_ratios must sum to a positive value.")

        return {
            resname: ratio / ratio_sum
            for resname, ratio in ratios.items()
        }

    def _target_area_per_lipid(self, normalized_ratios):
        """Return explicit or composition-estimated area per leaflet molecule."""

        if self.spec.area_per_lipid is not None:
            target_area = float(self.spec.area_per_lipid)
            if target_area <= 0:
                raise ValueError("area_per_lipid must be positive.")
            return target_area

        area_by_resname = DEFAULT_LIPID_AREAS.copy()
        if self.spec.lipid_area_overrides is not None:
            area_by_resname.update(
                {
                    resname.upper(): float(area)
                    for resname, area in self.spec.lipid_area_overrides.items()
                }
            )

        missing = [
            resname for resname in normalized_ratios
            if resname not in area_by_resname
        ]
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                "No default area-per-lipid is available for "
                f"{missing_names}. Pass area_per_lipid=... for the whole "
                "mixture, or lipid_area_overrides={...} for those residues."
            )

        target_area = sum(
            normalized_ratios[resname] * area_by_resname[resname]
            for resname in normalized_ratios
        )
        if target_area <= 0:
            raise ValueError("Estimated area per lipid must be positive.")
        return target_area

    def _available_leaflet_area(self):
        """Return membrane area available to lipids in one leaflet."""

        box_x, box_y = self.spec.box_xy
        box_area = float(box_x * box_y)

        if not self.spec.account_for_protein_footprint:
            return box_area

        return max(box_area - self._protein_footprint_area(), 0.0)

    def _protein_footprint_area(self):
        """Estimate XY area excluded by membrane-core protein atoms."""

        protein_xy = self._protein_xy_coords()
        if len(protein_xy) == 0:
            return 0.0

        spacing = float(self.spec.protein_footprint_sample_spacing)
        if spacing <= 0:
            raise ValueError("protein_footprint_sample_spacing must be positive.")

        box_x, box_y = self.spec.box_xy
        nx = max(1, int(np.ceil(box_x / spacing)))
        ny = max(1, int(np.ceil(box_y / spacing)))
        cell_area = (box_x / nx) * (box_y / ny)

        xs = np.linspace(
            -box_x / 2 + box_x / (2 * nx),
            box_x / 2 - box_x / (2 * nx),
            nx,
        )
        ys = np.linspace(
            -box_y / 2 + box_y / (2 * ny),
            box_y / 2 - box_y / (2 * ny),
            ny,
        )
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        sample_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        tree = KDTree(protein_xy)
        excluded_counts = tree.query_ball_point(
            sample_points,
            r=float(self.spec.protein_exclusion_radius),
            return_length=True,
        )
        return float(np.count_nonzero(excluded_counts) * cell_area)

    @staticmethod
    def _counts_from_ratios(total_count, normalized):
        """Convert a total leaflet count into integer residue counts."""

        counts = {
            resname: int(np.floor(total_count * ratio))
            for resname, ratio in normalized.items()
        }
        remainder = total_count - sum(counts.values())

        if remainder <= 0:
            return counts

        fractional = sorted(
            (
                (total_count * ratio - counts[resname], resname)
                for resname, ratio in normalized.items()
            ),
            reverse=True,
        )

        for _, resname in fractional[:remainder]:
            counts[resname] += 1

        return counts

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
        self._placed_membrane_coords_cache = []

        if self.packed_head_positions is not None:
            self._add_lipids_from_packed_heads(
                lipid_chain,
                sterol_chain,
                residue_numbers,
            )

            if len(lipid_chain) > 0:
                self.model.add(lipid_chain)
                self.topology.generate(lipid_chain)
    
            if len(sterol_chain) > 0:
                self.model.add(sterol_chain)
                self.topology.generate(sterol_chain)
    
            return

        for resname, coord in self._leaflet_positions_for_composition(
            self.upper_leaflet,
            leaflet_sign=1,
        ):
            self._add_membrane_residue(
                resname,
                coord,
                lipid_chain,
                sterol_chain,
                residue_numbers,
                leaflet_sign=1,
            )

        for resname, coord in self._leaflet_positions_for_composition(
            self.lower_leaflet,
            leaflet_sign=-1,
        ):
            self._add_membrane_residue(
                resname,
                coord,
                lipid_chain,
                sterol_chain,
                residue_numbers,
                leaflet_sign=-1,
            )

        if len(lipid_chain) > 0:
            self.model.add(lipid_chain)
            self.topology.generate(lipid_chain)

        if len(sterol_chain) > 0:
            self.model.add(sterol_chain)
            self.topology.generate(sterol_chain)

    def solvate(self):
        """Add water around the protein-membrane system."""
    
        solvator = Solvator(self.model)
    
        if self.spec.solvation_box is None:
            box_dims = (
                self.spec.box_xy[0],
                self.spec.box_xy[1],
                self.spec.box_z,
            )
        else:
            box_dims = self.spec.solvation_box
    
        water_chains = solvator.solvate(
            box_type="ortho",
            box_dims=box_dims,
            orient_coords=False,
            remove_existing_water=True,
            remove_existing_ions=False,
            solvcut=self.spec.water_solvcut,
        )
    
        if self.spec.remove_membrane_core_waters:
            self._remove_waters_in_membrane_core()
    
        return water_chains

    def _remove_waters_in_membrane_core(self):
        """Optionally remove waters in the membrane core for CHARMM-GUI-style cleanup."""
    
        removed = 0
        z_cutoff = float(self.spec.membrane_core_z)
    
        for chain in list(self.model):
            if getattr(chain, "chain_type", None) != "Solvent":
                continue
    
            for residue in list(chain):
                oxygen = None
    
                for atom in residue.get_atoms():
                    atom_name = atom.get_name().strip().upper()
                    if atom_name in {"OH2", "O"}:
                        oxygen = atom
                        break
    
                if oxygen is None:
                    continue
    
                if abs(float(oxygen.coord[2])) < z_cutoff:
                    chain.detach_child(residue.id)
                    removed += 1
    
            if len(chain) == 0:
                self.model.detach_child(chain.id)
    
        return removed

    def add_ions(self):
        """Replace waters with ions."""

        solvator = Solvator(self.model)
        solvator.add_ions(
            concentration=self.spec.salt_concentration,
            cation=self.spec.cation,
            anion=self.spec.anion,
        )

    def _load_charmm_gui_head_positions(self, crd_path):
        """Load CHARMM-GUI packed lipid head positions from step3_packing_head.crd."""

        crd_path = Path(crd_path)
        if not crd_path.exists():
            raise FileNotFoundError(f"CHARMM-GUI head CRD not found: {crd_path}")

        positions = []

        with open(crd_path, "r", encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("*")
            ]

        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue

            resname = fields[2].upper()
            coord = np.array(
                [float(fields[4]), float(fields[5]), float(fields[6])],
                dtype=float,
            )
            leaflet_sign = 1 if coord[2] >= 0 else -1

            positions.append(
                PackedHeadPosition(
                    resname=resname,
                    coord=coord,
                    leaflet_sign=leaflet_sign,
                )
            )
            
        if self.spec.align_head_positions_to_protein_xy:
            self._align_packed_heads_to_protein_xy(positions)

        return positions

    def _align_packed_heads_to_protein_xy(self, positions):
        """Translate packed head XY positions to the current protein XY center."""

        protein_coords = np.array(
            [
                atom.coord
                for chain in self.model
                if getattr(chain, "chain_type", None)
                not in {"Lipid", "Sterol", "Solvent", "Ion"}
                for atom in chain.get_atoms()
            ],
            dtype=float,
        )

        if len(protein_coords) == 0:
            return

        head_coords = np.array([position.coord for position in positions], dtype=float)

        protein_xy_center = protein_coords[:, :2].mean(axis=0)
        head_xy_center = head_coords[:, :2].mean(axis=0)
        shift_xy = protein_xy_center - head_xy_center

        for position in positions:
            position.coord[:2] = position.coord[:2] + shift_xy
    
    @staticmethod
    def _charmm_gui_head_z(resname: str):
        """Return CHARMM-GUI-style target head z for supported lipids."""

        resname = resname.upper()

        if resname == "POPC":
            return 19.0

        if resname == "CHL1":
            return 18.0

        return 19.0

    def _add_lipids_from_packed_heads(
        self,
        lipid_chain: Lipid,
        sterol_chain: Sterol,
        residue_numbers: dict[str, int],
    ):
        """Create membrane residues at CHARMM-GUI packed head positions."""

        expected = self._expected_head_position_counts()
        observed = {}

        for packed in self.packed_head_positions:
            leaflet = "upper" if packed.leaflet_sign > 0 else "lower"
            key = (leaflet, packed.resname)
            observed[key] = observed.get(key, 0) + 1

            target = packed.coord.copy()
            target[2] = packed.leaflet_sign * self._charmm_gui_head_z(packed.resname)

            self._add_membrane_residue(
                packed.resname,
                target,
                lipid_chain,
                sterol_chain,
                residue_numbers,
                leaflet_sign=packed.leaflet_sign,
            )

        if observed != expected:
            raise ValueError(
                "CHARMM-GUI packed head counts do not match MembraneSpec. "
                f"Expected {expected}, observed {observed}."
            )

    def _expected_head_position_counts(self):
        """Return expected CHARMM-GUI packed head counts by leaflet and resname."""

        expected = {}

        for resname, count in self.upper_leaflet.items():
            expected[("upper", resname.upper())] = count

        for resname, count in self.lower_leaflet.items():
            expected[("lower", resname.upper())] = count

        return expected
    
    def _add_membrane_residue(
        self,
        resname: str,
        center: np.ndarray,
        lipid_chain: Lipid,
        sterol_chain: Sterol,
        residue_numbers: dict[str, int],
        leaflet_sign: int,
    ):
        chain_type = self._chain_type_for_resname(resname)
        chain = sterol_chain if chain_type == "Sterol" else lipid_chain
        segid = "CHOL" if chain_type == "Sterol" else "MEMB"
        resseq = residue_numbers[chain_type]

        residue = self._create_residue(
            resname,
            chain_type,
            resseq,
            segid,
            center,
            leaflet_sign,
        )

        if self.spec.clash_check:
            residue = self._replace_with_nonclashing_residue(
                resname,
                chain_type,
                resseq,
                segid,
                center,
                leaflet_sign,
                residue,
            )
        
        chain.add(residue)
        
        if hasattr(self, "_placed_membrane_coords_cache"):
            coords = self._residue_heavy_atom_coords(residue)
            if len(coords) > 0:
                self._placed_membrane_coords_cache.append(coords)
                
        residue_numbers[chain_type] += 1
        return residue

    def _create_residue(
        self,
        resname: str,
        chain_type: str,
        resseq: int,
        segid: str,
        center: np.ndarray,
        leaflet_sign: int,
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

        if self.lipid_library is not None:
            conformer = self.lipid_library.random_conformer(resname, self.rng)
            self._apply_conformer_coords(residue, conformer)
            self._orient_library_residue_for_leaflet(residue, chain_type, leaflet_sign)
            self._random_rotate_residue_z(residue)
            self._translate_head_to_point(residue, chain_type, center)
        else:
            self._orient_residue_for_leaflet(residue, chain_type, leaflet_sign)
            self._scale_residue_z(residue, self.spec.lipid_z_scale)
            self._random_rotate_residue_z(residue)
            self._translate_residue_to_center(residue, center)
        
        return residue

    def _leaflet_positions_for_composition(self, leaflet_counts, leaflet_sign: int):
        """Return mixed lipid/sterol head positions for one leaflet."""

        total_count = sum(leaflet_counts.values())
        coords = self._leaflet_grid(total_count, z=0.0)

        resnames = []
        for resname, count in leaflet_counts.items():
            resnames.extend([resname.upper()] * count)

        self.rng.shuffle(resnames)

        positions = []
        for resname, coord in zip(resnames, coords):
            target = coord.copy()
            target[2] = leaflet_sign * self._charmm_gui_head_z(resname)
            positions.append((resname, target))

        return positions

    def _leaflet_grid(self, count: int, z: float):
        """Return jittered XY coordinates spread across one leaflet."""

        x_len, y_len = self.spec.box_xy
        protein_xy = self._protein_xy_coords()

        n_side = int(np.ceil(np.sqrt(count * 1.8)))

        while True:
            spacing_x = x_len / n_side
            spacing_y = y_len / n_side

            candidates = []

            for i in range(n_side):
                for j in range(n_side):
                    x = -x_len / 2 + (i + 0.5) * spacing_x
                    y = -y_len / 2 + (j + 0.5) * spacing_y

                    if j % 2 == 1:
                        x += 0.5 * spacing_x

                    x += self.rng.uniform(-0.25, 0.25) * spacing_x
                    y += self.rng.uniform(-0.25, 0.25) * spacing_y

                    if x < -x_len / 2 or x > x_len / 2:
                        continue
                    if y < -y_len / 2 or y > y_len / 2:
                        continue

                    point_xy = np.array([x, y], dtype=float)

                    if self._too_close_to_protein(point_xy, protein_xy):
                        continue

                    candidates.append(np.array([x, y, z], dtype=float))

            if len(candidates) >= count:
                break

            n_side += 1

            if n_side > 200:
                raise ValueError(
                    f"Could only place {len(candidates)} of {count} membrane residues. "
                    "Increase box_xy or reduce protein_exclusion_radius."
                )

        selected = self.rng.choice(len(candidates), size=count, replace=False)
        return [candidates[i] for i in selected]

    def _protein_xy_coords(self):
        """Return membrane-core protein XY coordinates for footprint exclusion."""

        coords = []
        z_cutoff = float(getattr(self.spec, "membrane_core_z", self.spec.lipid_z))

        for chain in self.model:
            if getattr(chain, "chain_type", None) in {"Lipid", "Sterol", "Solvent", "Ion"}:
                continue

            for atom in chain.get_atoms():
                if abs(float(atom.coord[2])) > z_cutoff:
                    continue

                coords.append(atom.coord[:2])

        if not coords:
            return np.empty((0, 2))

        return np.array(coords, dtype=float)

    def _too_close_to_protein(self, point_xy, protein_xy):
        """Check whether an XY membrane placement point overlaps protein footprint."""

        if len(protein_xy) == 0:
            return False

        distances = np.linalg.norm(protein_xy - point_xy, axis=1)
        return np.any(distances < self.spec.protein_exclusion_radius)

    def _has_sterols(self):
        resnames = set(self.upper_leaflet) | set(self.lower_leaflet)
        return any(self._is_sterol(resname) for resname in resnames)

    @staticmethod
    def _chain_type_for_resname(resname: str):
        return "Sterol" if MembraneBuilder._is_sterol(resname) else "Lipid"

    @staticmethod
    def _is_sterol(resname: str):
        return resname.upper() in STEROL_RESNAMES

    @staticmethod
    def _orient_residue_for_leaflet(residue, chain_type: str, leaflet_sign: int):
        """Orient one membrane residue along z with head/OH pointing outward."""

        atoms = list(residue.get_atoms())
        if len(atoms) < 3:
            return

        coords = np.array([atom.coord for atom in atoms], dtype=float)
        center = coords.mean(axis=0)
        centered = coords - center

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal_axis = vh[0]

        rotation = MembraneBuilder._rotation_matrix_from_vectors(
            principal_axis,
            np.array([0.0, 0.0, 1.0]),
        )

        rotated = centered @ rotation.T
        for atom, coord in zip(atoms, rotated + center):
            atom.coord = coord

        head_atoms = MembraneBuilder._head_atoms(residue, chain_type)
        if not head_atoms:
            return

        head_z = np.mean([atom.coord[2] for atom in head_atoms])
        body_z = np.mean(
            [
                atom.coord[2]
                for atom in atoms
                if atom not in head_atoms
            ]
        )

        if (head_z - body_z) * leaflet_sign < 0:
            MembraneBuilder._flip_lipid_z(residue)

    @staticmethod
    def _head_atoms(residue, chain_type: str):
        """Return atoms that define the outward-facing lipid head or sterol OH end."""

        atoms = list(residue.get_atoms())

        if chain_type == "Sterol":
            oxygen_atoms = [
                atom for atom in atoms
                if atom.get_name().strip().upper().startswith("O")
            ]
            return oxygen_atoms

        if chain_type == "Lipid":
            phosphate_atoms = [
                atom for atom in atoms
                if atom.get_name().strip().upper() == "P"
            ]
            if phosphate_atoms:
                return phosphate_atoms

        head_names = {
            "P", "N",
            "O11", "O12", "O13", "O14",
            "C11", "C12", "C13", "C14", "C15",
        }

        return [
            atom for atom in atoms
            if atom.get_name().strip().upper() in head_names
        ]

    @staticmethod
    def _rotation_matrix_from_vectors(source, target):
        """Return rotation matrix that aligns source vector to target vector."""

        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        source_norm = np.linalg.norm(source)
        target_norm = np.linalg.norm(target)

        if source_norm == 0 or target_norm == 0:
            return np.eye(3)

        source = source / source_norm
        target = target / target_norm

        cross = np.cross(source, target)
        dot = np.dot(source, target)

        if np.isclose(dot, 1.0):
            return np.eye(3)

        if np.isclose(dot, -1.0):
            axis = np.array([1.0, 0.0, 0.0])
            if np.isclose(abs(source[0]), 1.0):
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - source * np.dot(source, axis)
            axis = axis / np.linalg.norm(axis)
            return -np.eye(3) + 2.0 * np.outer(axis, axis)

        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )

        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / np.dot(cross, cross))

    @staticmethod
    def _scale_residue_z(residue, scale: float):
        """Scale residue thickness along z around its own center."""

        atoms = list(residue.get_atoms())
        if not atoms:
            return

        coords = np.array([atom.coord for atom in atoms], dtype=float)
        center_z = coords[:, 2].mean()

        for atom in atoms:
            atom.coord[2] = center_z + (atom.coord[2] - center_z) * scale

    def _random_rotate_residue_z(self, residue):
        """Randomly rotate one residue around z at its own center."""

        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation = np.array(
            [
                [cos_angle, -sin_angle, 0.0],
                [sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        atoms = list(residue.get_atoms())
        if not atoms:
            return

        coords = np.array([atom.coord for atom in atoms], dtype=float)
        center = coords.mean(axis=0)

        for atom in atoms:
            atom.coord = (rotation @ (atom.coord - center)) + center

    @staticmethod
    def _apply_conformer_coords(residue, conformer_coords):
        """Replace topology-template coordinates with lipid-library coordinates."""

        missing = []
        for atom in residue.get_atoms():
            atom_name = atom.get_name().strip()
            if atom_name not in conformer_coords:
                missing.append(atom_name)
                continue
            atom.coord = conformer_coords[atom_name].copy()

        if missing:
            raise ValueError(
                f"Lipid conformer missing {len(missing)} atoms for {residue.resname}: "
                f"{missing[:10]}"
            )

    @staticmethod
    def _orient_library_residue_for_leaflet(residue, chain_type: str, leaflet_sign: int):
        """Orient CHARMM-GUI library conformer with head group facing outward."""

        head_atoms = MembraneBuilder._head_atoms(residue, chain_type)
        tail_atoms = MembraneBuilder._tail_atoms(residue, chain_type)

        if not head_atoms or not tail_atoms:
            return

        head_z = np.mean([atom.coord[2] for atom in head_atoms])
        tail_z = np.mean([atom.coord[2] for atom in tail_atoms])

        if (head_z - tail_z) * leaflet_sign < 0:
            MembraneBuilder._flip_lipid_z(residue)

    @staticmethod
    def _tail_atoms(residue, chain_type: str):
        """Return atoms that define the inward-facing lipid tail/sterol tail end."""

        atoms = list(residue.get_atoms())

        if chain_type == "Sterol":
            tail_names = {"C26", "C27"}
            return [
                atom for atom in atoms
                if atom.get_name().strip().upper() in tail_names
            ]

        tail_names = {
            "C216", "C217", "C218",
            "C316", "C317", "C318",
            "C2", "C3",
        }

        return [
            atom for atom in atoms
            if atom.get_name().strip().upper() in tail_names
        ]

    @staticmethod
    def _translate_head_to_point(residue, chain_type: str, target_point):
        """Translate residue so its head group is centered on the leaflet point."""

        head_atoms = MembraneBuilder._head_atoms(residue, chain_type)
        if not head_atoms:
            MembraneBuilder._translate_residue_to_center(residue, target_point)
            return

        head_center = np.mean([atom.coord for atom in head_atoms], axis=0)
        shift = target_point - head_center

        for atom in residue.get_atoms():
            atom.coord = atom.coord + shift

    def _replace_with_nonclashing_residue(
        self,
        resname: str,
        chain_type: str,
        resseq: int,
        segid: str,
        center: np.ndarray,
        leaflet_sign: int,
        residue,
    ):
        """Retry conformers if the first placed residue cannot be repacked."""

        self._repack_residue_if_needed(residue, chain_type)
        if not self._has_bad_contacts(residue, chain_type):
            return residue

        for _ in range(self.spec.max_repack_attempts):
            candidate = self._create_residue(
                resname,
                chain_type,
                resseq,
                segid,
                center,
                leaflet_sign,
            )
            self._repack_residue_if_needed(candidate, chain_type)

            if not self._has_bad_contacts(candidate, chain_type):
                return candidate

        return residue
        
    def _repack_residue_if_needed(self, residue, chain_type: str):
        """Try simple CHARMM-GUI-like z-rotations and XY shifts to reduce clashes."""

        if not self._has_bad_contacts(residue, chain_type):
            return

        original_coords = {
            atom: atom.coord.copy()
            for atom in residue.get_atoms()
        }

        for _ in range(self.spec.max_repack_attempts):
            for atom, coord in original_coords.items():
                atom.coord = coord.copy()

            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            self._rotate_residue_z_about_center(residue, angle)

            if not self._has_bad_contacts(residue, chain_type):
                return

            for dx, dy in (
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (0.71, 0.71),
                (0.71, -0.71),
                (-0.71, 0.71),
                (-0.71, -0.71),
            ):
                for atom, coord in original_coords.items():
                    atom.coord = coord.copy()

                self._rotate_residue_z_about_center(residue, angle)
                self._translate_residue_xy(residue, dx, dy)

                if not self._has_bad_contacts(residue, chain_type):
                    return

        for atom, coord in original_coords.items():
            atom.coord = coord.copy()

    def _has_bad_contacts(self, residue, chain_type: str):
        """Return True if residue has heavy-atom contacts that are too close."""

        residue_coords = self._residue_heavy_atom_coords(residue)
        if len(residue_coords) == 0:
            return False

        protein_coords = self._protein_heavy_atom_coords()
        if len(protein_coords) > 0:
            distances = np.linalg.norm(
                residue_coords[:, None, :] - protein_coords[None, :, :],
                axis=2,
            )
            if np.any(distances < self.spec.protein_clash_cutoff):
                return True

        membrane_coords = self._placed_membrane_heavy_atom_coords()
        if len(membrane_coords) > 0:
            distances = np.linalg.norm(
                residue_coords[:, None, :] - membrane_coords[None, :, :],
                axis=2,
            )
            if np.any(distances < self.spec.membrane_clash_cutoff):
                return True

        return False

    def _protein_heavy_atom_coords(self):
        """Return heavy atom coordinates for non-membrane/non-solvent chains."""

        coords = []

        for chain in self.model:
            if getattr(chain, "chain_type", None) in {"Lipid", "Sterol", "Solvent", "Ion"}:
                continue

            for atom in chain.get_atoms():
                if atom.element == "H":
                    continue
                coords.append(atom.coord)

        if not coords:
            return np.empty((0, 3))

        return np.array(coords, dtype=float)

    def _placed_membrane_heavy_atom_coords(self):
        """Return heavy atom coords for membrane residues already placed."""
    
        coords = []
    
        if hasattr(self, "_placed_membrane_coords_cache"):
            coords.extend(self._placed_membrane_coords_cache)
    
        for chain in self.model:
            if getattr(chain, "chain_type", None) not in {"Lipid", "Sterol"}:
                continue
    
            for residue in chain:
                residue_coords = self._residue_heavy_atom_coords(residue)
                if len(residue_coords) > 0:
                    coords.append(residue_coords)
    
        if not coords:
            return np.empty((0, 3), dtype=float)
    
        return np.vstack(coords)

    @staticmethod
    def _residue_heavy_atom_coords(residue):
        """Return heavy atom coordinates for one residue."""

        coords = []

        for atom in residue.get_atoms():
            if atom.element == "H":
                continue
            coords.append(atom.coord)

        if not coords:
            return np.empty((0, 3))

        return np.array(coords, dtype=float)

    @staticmethod
    def _rotate_residue_z_about_center(residue, angle: float):
        """Rotate residue around z at its own center."""

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotation = np.array(
            [
                [cos_angle, -sin_angle, 0.0],
                [sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        atoms = list(residue.get_atoms())
        if not atoms:
            return

        center = np.mean([atom.coord for atom in atoms], axis=0)

        for atom in atoms:
            atom.coord = (rotation @ (atom.coord - center)) + center

    @staticmethod
    def _translate_residue_xy(residue, dx: float, dy: float):
        """Translate residue in the membrane plane."""

        shift = np.array([dx, dy, 0.0], dtype=float)

        for atom in residue.get_atoms():
            atom.coord = atom.coord + shift
    
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

def apply_charmm_gui_orientation(model: Model, orient_inp: str):
    """Apply CHARMM-GUI step2_orient.inp rotation/translation to a model."""
    
    rotation, translation = read_charmm_gui_orientation(orient_inp)
    
    for atom in model.get_atoms():
        atom.coord = rotation @ atom.coord + translation
    
    return model


def read_charmm_gui_orientation(orient_inp: str):
    """Read rotation matrix and translation vector from CHARMM-GUI step2_orient.inp."""
    
    with open(orient_inp, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    
    rotation = None
    translation = None
    
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
    
        if stripped.startswith("coor rotate matrix"):
            matrix_rows = []
            for row in lines[i + 1 : i + 4]:
                matrix_rows.append([float(value) for value in row.split()])
            rotation = np.array(matrix_rows, dtype=float)
    
        if stripped.startswith("coor trans"):
            fields = line.split()
            values = {}
    
            for idx, field in enumerate(fields):
                key = field.lower()
                if key in {"xdir", "ydir", "zdir"} and idx + 1 < len(fields):
                    values[key] = float(fields[idx + 1])
    
            translation = np.array(
                [
                    values.get("xdir", 0.0),
                    values.get("ydir", 0.0),
                    values.get("zdir", 0.0),
                ],
                dtype=float,
            )
    
    if rotation is None:
        raise ValueError(f"No CHARMM-GUI rotation matrix found in {orient_inp}")
    
    if translation is None:
        translation = np.zeros(3, dtype=float)
    
    return rotation, translation

def build_membrane_system(model: Model, spec: MembraneSpec) -> MembraneBuildResult:
    """Convenience function for one-call membrane building."""

    builder = MembraneBuilder(model, spec)
    return builder.build()




