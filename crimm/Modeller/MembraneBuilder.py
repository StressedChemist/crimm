from __future__ import annotations

"""Build protein-membrane systems using native crimm objects."""

from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
import json
import tarfile
import warnings

import numpy as np
from scipy.spatial import KDTree

from crimm.Modeller.Solvator import Solvator, wrap_periodic
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.StructEntities.Chain import Lipid, Sterol
from crimm.StructEntities.Model import Model


STEROL_RESNAMES = frozenset({"CHL1", "ERG", "SITO", "STIG", "CAMP", "GSITO", "GCAMP",
                            "GSTIG", "PGSITO", "PGCAMP", "PGSTIG"})

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
    "DSPC": 64.0,
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
    remove_membrane_core_waters: bool = True
    membrane_core_z: float = 15.0
    salt_concentration: float = 0.15
    cation: str = "POT"
    anion: str = "CLA"
    upper_lipid_ratios: dict[str, float] | None = None
    lower_lipid_ratios: dict[str, float] | None = None
    upper_lipid_area_overrides: dict[str, float] | None = None
    lower_lipid_area_overrides: dict[str, float] | None = None
    preset: str | None = None
    lipid_z_offsets: dict[str, float] | None = None
    initial_area_scale: float = 1.0

    @classmethod
    def from_preset(cls, name, *, box_xy, box_z, cholesterol_percent=None, **kwargs):
        """Use a published OPRLM composition without changing protein orientation."""
        from crimm.Modeller.MembranePresets import get_membrane_preset

        record = get_membrane_preset(name, cholesterol_percent)
        if record.get("packing_warning"):
            warnings.warn(record["packing_warning"], UserWarning, stacklevel=2)
        composition_keys = {"upper_leaflet", "lower_leaflet", "lipid_ratios",
                            "upper_lipid_ratios", "lower_lipid_ratios", "preset"}
        if composition_keys & kwargs.keys():
            raise ValueError("Use MembraneSpec directly for custom compositions or explicit counts.")
        settings = dict(upper_lipid_ratios=record["upper_ratios"],
                        lower_lipid_ratios=record["lower_ratios"], preset=name)
        if record["kind"] == "biological":
            settings["lipid_z_offsets"] = record.get("head_offsets_A")
            aliases = record["lipid_aliases"]
            for side, index in (("upper", 0), ("lower", 1)):
                areas = {
                    aliases.get(lipid, lipid): values[index]
                    for lipid, values in record["reference_areas_A2"].items() if values[index] is not None
                }
                counts = record[f"{side}_ratios"]
                reference_area = np.prod(record["reference_box_A"][:2])
                scale = reference_area / sum(count * areas[lipid] for lipid, count in counts.items())
                settings[f"{side}_lipid_area_overrides"] = {lipid: area * scale for lipid, area in areas.items()}
            settings["lipid_library_path"] = str(files("crimm.Data").joinpath("oprlm_lipids.npz"))
        settings.update(kwargs)
        return cls(box_xy=box_xy, box_z=box_z, **settings)


@dataclass
class MembraneBuildResult:
    """Summary of what was built."""

    model: Model
    lipid_count: int
    sterol_count: int
    water_count: int | None = None
    ion_count: int | None = None
    leaflet_composition: dict | None = None


class LipidConformerLibrary:
    """Load compact lipid coordinates or a legacy CHARMM-GUI archive once."""

    def __init__(self, archive_path=None):
        self.archive_path = (
            files("crimm.Data").joinpath("lipid_conformers.npz")
            if archive_path is None else Path(archive_path)
        )
        if not self.archive_path.is_file():
            raise FileNotFoundError(f"Lipid library not found: {self.archive_path}")

        self._atoms_by_resname = {}
        self._coords_by_resname = {}
        self._members_by_resname = {}
        self.templates = {}
        self._rtf_blocks = {}
        self._parameter_presets = set()
        with self.archive_path.open("rb") as handle:
            if self.archive_path.name.endswith(".npz"):
                with np.load(handle, allow_pickle=False) as data:
                    if "_templates" in data:
                        self.templates = json.loads(str(data["_templates"]))
                        self._rtf_blocks = {"Lipid": str(data["_lipid_rtf"]), "Sterol": str(data["_sterol_rtf"])}
                        self._parameter_presets = {key.removeprefix("_parameters_") for key in data.files
                                                   if key.startswith("_parameters_")}
                    for key in data.files:
                        if key.endswith("_coords"):
                            resname = key.removesuffix("_coords")
                            scale = float(data.get(f"{resname}_scale", 1.0))
                            if not np.isfinite(scale) or scale <= 0:
                                raise ValueError("Invalid lipid coordinate scale")
                            self._atoms_by_resname[resname] = data[f"{resname}_atoms"].tolist()
                            self._coords_by_resname[resname] = data[key].astype(float) / scale
            else:
                with tarfile.open(fileobj=handle, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(".crd"):
                            continue
                        parts = member.name.split("/")
                        if len(parts) < 4:
                            continue
                        resname = parts[1].upper()
                        with archive.extractfile(member) as crd:
                            atoms, coords = self._read_crd(crd.read().decode("utf-8"), resname)
                        if resname in self._atoms_by_resname:
                            expected_atoms = self._atoms_by_resname[resname]
                            if set(atoms) != set(expected_atoms):
                                raise ValueError(f"Inconsistent lipid atom names for {resname}")
                            by_name = dict(zip(atoms, coords))
                            coords = np.array([by_name[name] for name in expected_atoms])
                        else:
                            self._atoms_by_resname[resname] = atoms
                        self._coords_by_resname.setdefault(resname, []).append(coords)
                        self._members_by_resname.setdefault(resname, []).append(member.name)
                self._coords_by_resname = {
                    resname: np.asarray(coords) for resname, coords in self._coords_by_resname.items()
                }

        if not self._coords_by_resname:
            raise ValueError("No lipid conformers found in library")
        for resname, coords in self._coords_by_resname.items():
            atoms = self._atoms_by_resname[resname]
            if (coords.ndim != 3 or coords.shape[1:] != (len(atoms), 3)
                    or len(coords) == 0 or not atoms or len(set(atoms)) != len(atoms)
                    or not np.isfinite(coords).all()):
                raise ValueError(f"Invalid lipid coordinates for {resname}")
            if resname in self.templates:
                valid = self.templates[resname]["valid_conformers"]
                if not valid or any(not isinstance(index, int) or not 0 <= index < len(coords) for index in valid):
                    raise ValueError(f"Invalid stereo-compliant conformer indices for {resname}.")

    def install_topology(self, topology, preset=None):
        """Register complete reference molecules through the existing loaders."""
        if not self.templates:
            return
        if preset not in self._parameter_presets:
            raise ValueError("The OPRLM lipid library requires a biological preset to select its reference parameters.")
        from crimm.IO.RTFParser import RTFParser
        from crimm.IO.PRMParser import categorize_lines, parse_line_dict
        from crimm.Modeller.TopoLoader import ParameterLoader, ResidueTopologySet, chain_type_def_lookup

        parameters = ParameterLoader()
        with self.archive_path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            parameters._raw_data_strings = str(data[f"_parameters_{preset}"]).splitlines()
        parameters.param_dict = parse_line_dict(categorize_lines(parameters._raw_data_strings))
        for chain_type, block in self._rtf_blocks.items():
            entity_type = chain_type_def_lookup[chain_type]
            definitions = ResidueTopologySet(entity_type, rtf_parser=RTFParser(rtf_block=block))
            for name, definition in definitions.res_defs.items():
                template = self.templates[name]
                if set(definition.atom_dict) != set(self._atoms_by_resname[name]):
                    raise ValueError(f"Reference topology and conformer atom names differ for {name}.")
                definition.membrane_head_atoms = set(template["head_atoms"])
                definition.membrane_tail_atoms = set(template["tail_atoms"])
                definition.membrane_template = template
            topology.res_def_dict[entity_type] = definitions
            topology.param_dict[entity_type] = parameters

    def random_conformer(self, resname: str, rng):
        resname = resname.upper()
        if resname not in self._coords_by_resname:
            raise ValueError(f"No lipid conformers found for {resname}")
        conformers = self._coords_by_resname[resname]
        valid = self.templates[resname]["valid_conformers"] if resname in self.templates else range(len(conformers))
        coords = conformers[valid[rng.integers(len(valid))]].copy()
        return dict(zip(self._atoms_by_resname[resname], coords))

    @staticmethod
    def _read_crd(text, resname):
        lines = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("*")]
        atom_count = int(lines[0][0])
        if len(lines) - 1 != atom_count:
            raise ValueError("Lipid CRD atom count does not match its coordinates")
        lipid_rows = [fields for fields in lines[1:] if fields[2].upper() == resname]
        by_name = {
            fields[3]: [float(value) for value in fields[4:7]]
            for fields in lipid_rows
        }
        if len(by_name) != len(lipid_rows):
            raise ValueError(f"Duplicate lipid atom names for {resname}")
        return list(by_name), np.array(list(by_name.values()))

@dataclass
class PackedHeadPosition:
    resname: str
    coord: np.ndarray
    leaflet_sign: int

class MembraneBuilder:
    """Build a protein embedded in a lipid membrane."""

    LIPID_TAILS = {
        "DMPC": {"C214", "C314"}, "DPPC": {"C216", "C316"},
        "DSPC": {"C218", "C318"}, "POPC": {"C218", "C316"},
        "DOPC": {"C218", "C318"}, "CHL1": {"C26", "C27"},
    }

    def __init__(self, model: Model, spec: MembraneSpec):
        self.model = model
        self.spec = spec
        self._box_dims = np.asarray(spec.solvation_box or (*spec.box_xy, spec.box_z), dtype=float)
        if self._box_dims.shape != (3,) or not np.isfinite(self._box_dims).all() or np.any(self._box_dims <= 0):
            raise ValueError("Membrane box dimensions must be finite and positive.")
        if not np.allclose(self._box_dims[:2], spec.box_xy, rtol=0, atol=1e-12):
            raise ValueError("solvation_box XY dimensions must match box_xy for a periodic membrane.")
        if not np.isfinite(spec.initial_area_scale) or spec.initial_area_scale < 1:
            raise ValueError("initial_area_scale must be finite and at least 1.")
        self.topology = getattr(model, "topology_loader", None) or TopologyGenerator()
        self.model.topology_loader = self.topology
        self.rng = np.random.default_rng(spec.random_seed)
        self.leaflet_composition = {}
        self.upper_leaflet = self._resolve_leaflet_counts("upper")
        self.lower_leaflet = self._resolve_leaflet_counts("lower")
        # Resolve nominal counts before expansion, without stretching any molecule.
        if spec.initial_area_scale != 1:
            self._box_dims[:2] *= np.sqrt(spec.initial_area_scale)
            self.spec = replace(spec, box_xy=tuple(self._box_dims[:2]),
                                solvation_box=tuple(self._box_dims) if spec.solvation_box is not None else None)
    
        self.lipid_library = None
        if spec.use_lipid_library:
            self.lipid_library = LipidConformerLibrary(spec.lipid_library_path)
            self.lipid_library.install_topology(self.topology, spec.preset)

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
            specific = self.spec.upper_lipid_ratios
        elif leaflet == "lower":
            explicit = self.spec.lower_leaflet
            specific = self.spec.lower_lipid_ratios
        else:
            raise ValueError(f"Unknown leaflet: {leaflet}")

        if explicit is not None:
            if specific is not None:
                raise ValueError(f"Cannot provide both counts and ratios for the {leaflet} leaflet.")
            values = self._lipid_values(explicit)
            if any(count != int(count) for count in values.values()):
                raise ValueError("Leaflet lipid counts must be integers.")
            counts = {name: int(count) for name, count in values.items() if count > 0}
            self._record_composition(leaflet, counts, None, None)
            return counts

        ratios_input = specific if specific is not None else self.spec.lipid_ratios
        if ratios_input is None:
            raise ValueError(
                "Either upper_leaflet/lower_leaflet counts or lipid_ratios must be provided."
            )

        ratios = self._normalized_lipid_ratios(ratios_input)
        target_area = self._target_area_per_lipid(ratios, leaflet)
        leaflet_area = self._available_leaflet_area()
        total_count = int(round(leaflet_area / target_area))

        if total_count <= 0:
            raise ValueError("Estimated leaflet lipid count must be positive.")

        counts = {name: count for name, count in self._counts_from_ratios(total_count, ratios).items() if count > 0}
        self._record_composition(leaflet, counts, ratios, target_area)
        return counts

    @staticmethod
    def _lipid_values(values):
        normalized = {}
        for name, value in values.items():
            name = name.strip().upper()
            if not name or name in normalized:
                raise ValueError("Lipid names must be nonempty and unique after normalization.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Lipid values must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError("Lipid values must be finite and non-negative.")
            normalized[name] = value
        return normalized

    def _record_composition(self, leaflet, counts, ratios, area):
        total = sum(counts.values())
        self.leaflet_composition[leaflet] = {
            "requested_percent": {name: 100 * ratio for name, ratio in ratios.items()} if ratios is not None else None,
            "counts": dict(counts),
            "realized_percent": {name: 100 * count / total for name, count in counts.items()} if total else {},
            "area_per_molecule": area,
        }

    def _normalized_lipid_ratios(self, values=None):
        """Return normalized lipid/sterol ratios from the membrane spec."""

        values = self.spec.lipid_ratios if values is None else values
        ratios = {name: value for name, value in self._lipid_values(values).items() if value > 0}

        ratio_sum = sum(ratios.values())
        if not np.isfinite(ratio_sum) or ratio_sum <= 0:
            raise ValueError("lipid_ratios must sum to a positive value.")

        return {
            resname: ratio / ratio_sum
            for resname, ratio in ratios.items()
        }

    def _target_area_per_lipid(self, normalized_ratios, leaflet=None):
        """Return explicit or composition-estimated area per leaflet molecule."""

        if self.spec.area_per_lipid is not None:
            target_area = float(self.spec.area_per_lipid)
            if not np.isfinite(target_area) or target_area <= 0:
                raise ValueError("area_per_lipid must be positive.")
            return target_area

        area_by_resname = DEFAULT_LIPID_AREAS.copy()
        if self.spec.lipid_area_overrides is not None:
            area_by_resname.update(
                {
                    resname: area
                    for resname, area in self._lipid_values(self.spec.lipid_area_overrides).items()
                }
            )
        specific_areas = getattr(self.spec, f"{leaflet}_lipid_area_overrides", None)
        if specific_areas is not None:
            area_by_resname.update(self._lipid_values(specific_areas))

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
        if (not np.isfinite(target_area) or target_area <= 0
                or any(area_by_resname[name] <= 0 for name in normalized_ratios)):
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

        box_xy = self._box_dims[:2]
        tree = KDTree(wrap_periodic(protein_xy, box_xy), boxsize=box_xy)
        excluded_counts = tree.query_ball_point(
            wrap_periodic(sample_points, box_xy),
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

        fractional = sorted(normalized, key=lambda name: total_count * normalized[name] - counts[name], reverse=True)
        for resname in fractional[:remainder]:
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
            leaflet_composition=self.leaflet_composition,
        )

    def prepare_topology(self):
        """Load lipid, sterol, water, and ion topology definitions."""

        self.topology.load_residue_definitions("Lipid")
        if self._has_sterols():
            self.topology.load_residue_definitions("Sterol")
        self.topology.load_residue_definitions("Solvent")

    def add_lipid_bilayer(self):
        """Create upper and lower leaflet lipid and sterol residues."""

        self.prepare_topology()
        requested = {name for leaf in (self.upper_leaflet, self.lower_leaflet) for name, count in leaf.items() if count}
        for resname in requested:
            chain_type = self._chain_type_for_resname(resname)
            definitions, parameters = self.topology.load_residue_definitions(chain_type)
            if resname not in definitions:
                raise ValueError(f"No membrane topology for {resname}.")
            if self.lipid_library is not None:
                if resname not in self.lipid_library._coords_by_resname:
                    raise ValueError(f"No lipid conformers for {resname}.")
                if set(self.lipid_library._atoms_by_resname[resname]) != set(definitions[resname].atom_dict):
                    raise ValueError(f"Lipid conformer atom names differ from the {resname} topology.")
            for definition in definitions[resname].atom_dict.values():
                if definition.atom_type not in parameters.param_dict["nonbonded"]:
                    raise ValueError(f"No nonbonded parameters for {resname}/{definition.atom_type}.")
            probe = Sterol("CHECK") if chain_type == "Sterol" else Lipid("CHECK")
            if self.lipid_library is not None:
                coordinates = dict(zip(self.lipid_library._atoms_by_resname[resname],
                                       self.lipid_library._coords_by_resname[resname][0]))
                probe.add(definitions[resname].create_residue_from_coord_dict(coordinates, 1, segid="CHECK"))
            else:
                probe.add(definitions[resname].create_residue(1, segid="CHECK"))
            self.topology.generate(probe, QUIET=True)
            if probe.topology.missing_param_dict:
                raise ValueError(f"Missing bonded parameters for {resname}: {list(probe.topology.missing_param_dict)}.")

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
            periodic=True,
        )
    
        if self.spec.remove_membrane_core_waters:
            self._remove_waters_in_membrane_core()
    
        return [chain for chain in water_chains if chain.parent is self.model]

    def _remove_waters_in_membrane_core(self):
        """Optionally remove waters in the membrane core for CHARMM-GUI-style cleanup."""
    
        removed = 0
        z_cutoff = float(self.spec.membrane_core_z)
        if not np.isfinite(z_cutoff) or z_cutoff <= 0:
            raise ValueError("membrane_core_z must be finite and positive.")
    
        for chain in list(self.model):
            if getattr(chain, "chain_type", None) != "Solvent":
                continue
            removed_from_chain = 0
            for residue in list(chain):
                if any(abs(float(atom.coord[2])) < z_cutoff for atom in residue.get_atoms()):
                    chain.detach_child(residue.id)
                    removed += 1
                    removed_from_chain += 1
    
            if len(chain) == 0:
                self.model.detach_child(chain.id)
            elif removed_from_chain:
                self.topology.generate_solvent(chain, "TIP3", QUIET=True)
    
        return removed

    def add_ions(self):
        """Replace waters with ions."""

        solvator = Solvator(self.model)
        if self.spec.preset == "G-OM":
            # The published reference uses 175 Ca2+ for 35 LPS molecules.
            count = 5 * (self.upper_leaflet.get("ECLIPA", 0) + self.lower_leaflet.get("ECLIPA", 0))
            existing = [chain for chain in self.model if getattr(chain, "source", None) == "membrane_counterions"]
            if existing:
                if sum(len(chain) for chain in existing) != count:
                    raise ValueError("The existing LPS counterion count differs from the preset.")
            else:
                waters = [residue for chain in self.model if chain.chain_type == "Solvent"
                          for residue in chain if residue["OH2"].coord[2] >= self.spec.membrane_core_z]
                solvator.add_ions_by_count({"CAL": count}, eligible_waters=waters, rng=self.rng)
        solvator.add_ions(
            concentration=self.spec.salt_concentration,
            cation=self.spec.cation,
            anion=self.spec.anion,
            rng=self.rng,
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
            coord[:2] *= np.sqrt(self.spec.initial_area_scale)
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

        if not positions:
            return

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
    
    def _charmm_gui_head_z(self, resname: str):
        """Return the requested leaflet headgroup z position."""

        height = float(self.spec.lipid_z) + (self.spec.lipid_z_offsets or {}).get(resname, 0.0)
        if not np.isfinite(height) or height <= 0:
            raise ValueError("Membrane lipid-head heights must be finite and positive.")
        return height

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

        if self.lipid_library is not None:
            conformer = self.lipid_library.random_conformer(resname, self.rng)
            residue = residue_defs[resname].create_residue_from_coord_dict(conformer, resseq=resseq, segid=segid)
        else:
            residue = residue_defs[resname].create_residue(resseq=resseq, segid=segid)
        if residue is None:
            raise ValueError(f"Could not create membrane residue {resname}")

        if self.lipid_library is not None:
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

        if self.lipid_library is not None and self.lipid_library.templates:
            positions.sort(key=lambda entry: len(self.lipid_library.templates[entry[0]]["atoms"]), reverse=True)
        return positions

    def _leaflet_grid(self, count: int, z: float):
        """Return regular periodic XY sites outside the protein footprint."""

        if count < 0:
            raise ValueError("Leaflet lipid count must be non-negative.")
        if count == 0:
            return []

        x_len, y_len = self.spec.box_xy
        protein_xy = self._protein_xy_coords()

        available_area = self._available_leaflet_area()
        if available_area <= 0:
            raise ValueError("The protein footprint leaves no membrane area.")
        n_side = int(np.ceil(np.sqrt(count * x_len * y_len / available_area)))

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

                    x = (x + x_len / 2) % x_len - x_len / 2

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

        delta = protein_xy - point_xy
        delta -= self._box_dims[:2] * np.rint(delta / self._box_dims[:2])
        distances = np.linalg.norm(delta, axis=1)
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
        definition = residue.topo_definition or (getattr(atoms[0].topo_definition, "parent_def", None) if atoms else None)
        reference_heads = getattr(definition, "membrane_head_atoms", None)
        if reference_heads is not None:
            return [atom for atom in atoms if atom.name in reference_heads]

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
        definition = residue.topo_definition or (getattr(atoms[0].topo_definition, "parent_def", None) if atoms else None)
        reference_tails = getattr(definition, "membrane_tail_atoms", None)
        if reference_tails is not None:
            return [atom for atom in atoms if atom.name in reference_tails]

        if chain_type == "Sterol":
            tail_names = {"C26", "C27"}
            return [
                atom for atom in atoms
                if atom.get_name().strip().upper() in tail_names
            ]

        tail_names = MembraneBuilder.LIPID_TAILS.get(residue.resname, {
            "C216", "C217", "C218",
            "C316", "C317", "C318",
            "C2", "C3",
        })

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

        raise ValueError(
            f"Could not place {resname} {resseq} without periodic clashes after "
            f"{self.spec.max_repack_attempts} conformer retries. "
            "Increase the membrane area or reduce the lipid count."
        )

    def _repack_residue_if_needed(self, residue, chain_type: str):
        """Try simple CHARMM-GUI-like z-rotations and XY shifts to reduce clashes."""

        if not self._has_bad_contacts(residue, chain_type):
            return

        original_coords = {
            atom: atom.coord.copy()
            for atom in residue.get_atoms()
        }

        shift = 2.0
        local_attempts = self.spec.max_repack_attempts
        if self.lipid_library is not None and self.lipid_library.templates:
            heads = self._head_atoms(residue, chain_type)
            counts = self.upper_leaflet if np.mean([atom.coord[2] for atom in heads]) > 0 else self.lower_leaflet
            shift = max(shift, np.sqrt(np.prod(self.spec.box_xy) / sum(counts.values())) / 2)
            # Large reference molecules need moves scaled to their head spacing.
            local_attempts = min(local_attempts, 20)
        diagonal = shift / np.sqrt(2) if shift != 2.0 else 1.42
        for _ in range(local_attempts):
            for atom, coord in original_coords.items():
                atom.coord = coord.copy()

            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            self._rotate_residue_z_about_head(residue, chain_type, angle)

            if not self._has_bad_contacts(residue, chain_type):
                return

            moves = [
                (shift, 0.0),
                (-shift, 0.0),
                (0.0, shift),
                (0.0, -shift),
                (diagonal, diagonal),
                (diagonal, -diagonal),
                (-diagonal, diagonal),
                (-diagonal, -diagonal),
            ]
            if self.lipid_library is not None and self.lipid_library.templates:
                moves.append(self.rng.uniform(-np.array(self.spec.box_xy) / 2, np.array(self.spec.box_xy) / 2))
            for dx, dy in moves:
                for atom, coord in original_coords.items():
                    atom.coord = coord.copy()

                self._rotate_residue_z_about_head(residue, chain_type, angle)
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
            tree = KDTree(wrap_periodic(protein_coords, self._box_dims), boxsize=self._box_dims)
            distances = tree.query(wrap_periodic(residue_coords, self._box_dims))[0]
            if np.any(distances < self.spec.protein_clash_cutoff):
                return True

        cache = getattr(self, "_placed_membrane_coords_cache", None)
        cacheable = cache is not None and not any(
            getattr(chain, "chain_type", None) in {"Lipid", "Sterol"} for chain in self.model)
        cached = getattr(self, "_membrane_contact_index", None)
        if cacheable and cached is not None and cached[0] is cache and cached[1] == len(cache):
            tree = cached[2]
        else:
            membrane_coords = self._placed_membrane_heavy_atom_coords()
            tree = (KDTree(wrap_periodic(membrane_coords, self._box_dims), boxsize=self._box_dims)
                    if len(membrane_coords) else None)
            self._membrane_contact_index = (cache, len(cache), tree) if cacheable else None
        if tree is not None:
            distances = tree.query(wrap_periodic(residue_coords, self._box_dims))[0]
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
    def _rotate_residue_z_about_head(residue, chain_type: str, angle: float):
        """Rotate residue around z without moving its head."""

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

        center = np.mean([
            atom.coord for atom in MembraneBuilder._head_atoms(residue, chain_type)
        ], axis=0)

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
        """Reverse the leaflet direction with a proper 180-degree X rotation."""
        atoms = list(residue.get_atoms())
        if not atoms:
            return
        coords = np.array([atom.coord for atom in atoms])
        center = coords.mean(axis=0)

        for atom in atoms:
            atom.coord[1:] = 2 * center[1:] - atom.coord[1:]

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
