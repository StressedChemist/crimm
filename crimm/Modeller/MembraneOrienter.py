"""Orient proteins with the native PPM3 positioning model.

PPM3 uses a DOPC profile and its own ionization model, not CHARMM charges.
The native optimizer returns a full rigid transform for the complete model.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from importlib.util import find_spec
from string import ascii_letters, digits
from threading import Lock

import numpy as np


PPM3_REVISION = "5784f088032e5b9fb35147815f67500b8cc94855"
PPM3_MEMBRANE_TYPES = (
    "PMm", "PMp", "PMf", "ERf", "ERm", "GOL", "LYS", "END", "VAC", "MOM", "MIM",
    "THp", "THb", "GnO", "GnI", "GpI", "ARC", "LPC", "MPC", "OPC", "EPC",
)

PROTEIN_CHAIN_TYPES = frozenset({"Polypeptide(L)", "Protein", "Chain", "Base Chain"})
PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS HSD HSE HSP ILE LEU LYS MET "
    "PHE PRO SER THR TRP TYR VAL".split()
)
TERMINAL_ATOMS = {
    "ACE": {"CAY", "CY", "OY"}, "ACED": {"CAY", "CY", "OY"},
    "ACP": {"CAY", "CY", "OY"}, "ACPD": {"CAY", "CY", "OY"},
    "CT2": {"NT"}, "CT3": {"NT", "CAT"},
}
_NATIVE_LOCK = Lock()  # ponytail: COMMON state serializes calls; use separate processes for throughput.
_NATIVE_ERRORS = {
    1: "invalid native input",
    2: "missing or malformed residue library",
    3: "unsupported residue or atom typing",
    4: "native capacity exceeded",
    5: "no favorable membrane insertion minimum",
    6: "undefined native geometry or rigid transform",
}


@dataclass(frozen=True)
class MembraneOrientation:
    """Native transform and positioning metrics.

    Coordinates use Å. Energy uses kcal/mol, and angles use degrees.
    The tilt is the optimizer angle, not a fitted transmembrane-axis angle.
    The model is the native classification before final optimization.
    Positioning uncertainty is Z-shift RMS for model 1 and thickness-range
    half-width for models 2 and 3. Native spreads are not confidence intervals.
    """

    rotation: np.ndarray
    translation: np.ndarray
    score: float
    membrane_thickness: float
    tilt: float
    positioning_uncertainty: float
    tilt_uncertainty: float
    model: int
    ph: float
    temperature: float
    profile: str = "DOPC"
    scored_atoms: str = "protein core heavy atoms"
    membrane_type: str = "OPC"
    native_version: str = "PPM3"
    source_revision: str = PPM3_REVISION
    equilibrium_thickness: float | None = None
    matching_thickness: float | None = None
    mismatch_coefficient: float | None = None
    topology: str = "in"
    topology_reference: dict | None = None


@lru_cache(maxsize=1)
def _native_function():
    spec = find_spec("crimm.Modeller._ppm3._native")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "The native PPM3 library is unavailable. "
            "Build it with 'python setup.py build_ext --inplace' or install a compiled wheel."
        )
    try:
        library = ctypes.CDLL(spec.origin)
    except OSError as exc:
        raise RuntimeError(f"The native PPM3 library did not load: {exc}") from exc
    function = library.ppm3_orient
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_char_p,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="F_CONTIGUOUS"),
        vector, vector, vector, ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None
    return function


@lru_cache(maxsize=1)
def _residue_atom_names():
    atom_names = {}
    for line in files("crimm.Modeller._ppm3").joinpath("res.lib").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "RES":
            resname = fields[1]
            atom_names[resname] = set()
        elif len(fields) == 2 and fields[1].isdigit():
            atom_names[resname].add(fields[0])
    return atom_names


class MembraneOrienter:
    """Use PPM3 without a Python scorer or fallback."""

    def __init__(self, model, ph: float = 7.5, temperature: float = 293.0, topology: str = "in",
                 membrane_type: str = "OPC"):
        self.model = model
        self.ph = float(ph)
        self.temperature = float(temperature)
        self.topology = topology
        self.membrane_type = membrane_type
        if membrane_type not in PPM3_MEMBRANE_TYPES:
            raise ValueError(f"PPM3 requires a named planar membrane code: {', '.join(PPM3_MEMBRANE_TYPES)}.")
        if not np.isfinite(self.ph) or not 0.0 <= self.ph <= 14.0:
            raise ValueError("PPM3 pH must be finite and between 0 and 14.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("PPM3 temperature must be finite and positive in kelvin.")
        if topology not in {"in", "out"}:
            raise ValueError("PPM3 topology must be 'in' or 'out'.")

    def orient(self, in_place: bool = True) -> MembraneOrientation:
        """Find a native orientation and optionally transform the complete model."""

        all_coords = np.array([
            atom.coord for residue in self.model.get_residues()
            for atom in residue.get_unpacked_list()
        ], dtype=np.float64)
        if (all_coords.size and (all_coords.ndim != 2 or all_coords.shape[1] != 3
                                 or not np.isfinite(all_coords).all())):
            raise ValueError("All retained model coordinates must contain three finite values per atom.")
        coords, names, residues, chains, residue_ids = self._protein_input()
        rotation = np.empty((3, 3), dtype=np.float64, order="F")
        translation = np.empty(3, dtype=np.float64)
        metrics = np.empty(6, dtype=np.float64)
        matching = np.empty(3, dtype=np.float64)
        status = ctypes.c_int()
        path = str(files("crimm.Modeller._ppm3").joinpath("res.lib")).encode()
        with _NATIVE_LOCK:
            _native_function()(
                len(coords), coords, names, residues, chains, residue_ids,
                path, len(path), self.ph, self.temperature, int(self.topology == "out"),
                self.membrane_type.encode("ascii"), rotation, translation, metrics, matching, ctypes.byref(status),
            )
        if status.value:
            reason = _NATIVE_ERRORS.get(status.value, "unknown native error")
            raise RuntimeError(f"PPM3 orientation failed: {reason} (status {status.value}).")
        if (not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.isfinite(metrics).all() or not np.isfinite(matching).all()
                or np.any(matching[:2] <= 0) or matching[2] < 0 or metrics[1] <= 0
                or metrics[5] not in (1, 2, 3)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5, rtol=0)):
            raise RuntimeError("PPM3 returned an invalid rigid transform or positioning metrics.")
        # Remove native REAL rounding from the rigid matrix, with the same native centroid.
        u, _, vt = np.linalg.svd(rotation)
        rigid_rotation = u @ vt
        translation += (rotation - rigid_rotation) @ coords[:10].mean(axis=0)
        reference = dict(self._reference)
        reference["final_z"] = float((rigid_rotation @ coords[0] + translation)[2])
        if abs(reference["final_z"]) <= 0.002:
            raise RuntimeError("PPM3 topology reference is within 0.002 Å of the membrane midplane.")
        result = MembraneOrientation(
            rigid_rotation, translation, *metrics[:5], int(metrics[5]), self.ph, self.temperature,
            membrane_type=self.membrane_type, equilibrium_thickness=float(matching[0]),
            matching_thickness=float(matching[1]), mismatch_coefficient=float(matching[2]),
            topology=self.topology, topology_reference=reference,
        )
        if in_place:
            self.apply_orientation(result)
        return result

    def apply_orientation(self, orientation: MembraneOrientation):
        """Apply the complete native transform to every retained atom."""

        atoms = [atom for residue in self.model.get_residues() for atom in residue.get_unpacked_list()]
        coords = np.array([atom.coord for atom in atoms], dtype=np.float64)
        transformed = coords @ orientation.rotation.T + orientation.translation
        if not np.isfinite(transformed).all():
            raise ValueError("The model contains invalid coordinates for the native transform.")
        for atom, coord in zip(atoms, transformed):
            atom.coord = coord

    def _protein_input(self):
        coords, names, residues, chains, residue_ids = [], [], [], [], []
        atom_names = _residue_atom_names()
        residue_number = 0
        self._reference = None
        protein_chains = [
            chain for chain in self.model
            if getattr(chain, "chain_type", None) in PROTEIN_CHAIN_TYPES | {None}
            and any(residue.resname.strip().upper() in PROTEIN_RESIDUES for residue in chain)
        ]
        if len(protein_chains) > 60:
            raise ValueError("PPM3 supports at most 60 protein chains.")
        for chain_id, chain in zip(ascii_letters + digits, protein_chains):
            previous_number = None
            sequence_id = 0
            protein_residues = [residue for residue in chain if not residue.id[0].strip()
                                or residue.resname.strip().upper() in PROTEIN_RESIDUES]
            for index, residue in enumerate(protein_residues):
                resname = residue.resname.strip().upper()
                if resname not in PROTEIN_RESIDUES:
                    raise ValueError(f"PPM3 does not support protein residue {resname} at {residue.full_id}.")
                if resname in {"HIS", "HSD", "HSP"}:
                    raise ValueError(
                        f"PPM3 does not support {resname} at {residue.full_id}. "
                        "The native neutral histidine model requires an explicit HSE identity."
                    )
                patch = getattr(getattr(residue, "topo_definition", None), "patch_with", None)
                if patch is not None and patch not in TERMINAL_ATOMS:
                    raise ValueError(f"PPM3 does not support patch {patch} at {residue.full_id}.")
                if ((patch in {"ACE", "ACED", "ACP", "ACPD"} and index != 0)
                        or (patch in {"CT2", "CT3"} and index != len(protein_residues) - 1)):
                    raise ValueError(f"PPM3 found a nonterminal cap patch {patch} at {residue.full_id}.")
                native_resname = "HIS" if resname == "HSE" else resname
                expected = atom_names[native_resname]
                all_atoms = list(residue.get_atoms())
                if resname == "HSE" and any(atom.name.strip().upper() == "HD1" for atom in all_atoms):
                    raise ValueError(f"HSE contains an incompatible HD1 proton at {residue.full_id}.")
                heavy_atoms = [
                    atom for atom in all_atoms
                    if getattr(atom, "element", "").strip().upper() not in {"H", "D"}
                    and not atom.name.strip().upper().startswith(("H", "D"))
                ]
                present = {atom.name.strip().upper() for atom in heavy_atoms}
                if expected - present:
                    raise ValueError(
                        f"PPM3 requires complete protein core atoms at {residue.full_id}: "
                        f"missing {', '.join(sorted(expected - present))}."
                    )
                if self._reference is None:
                    self._reference = {"chain": getattr(chain, "id", chain_id), "residue": residue.id,
                                       "atom": "N", "kind": "first available backbone N"}
                allowed = TERMINAL_ATOMS.get(patch, set())
                if index == len(protein_residues) - 1:
                    allowed = allowed | {"OXT"}
                unsupported = present - expected - allowed
                if unsupported:
                    raise ValueError(
                        f"PPM3 found unsupported protein atoms at {residue.full_id}: "
                        f"{', '.join(sorted(unsupported))}."
                    )
                residue_number += 1
                if residue_number > 20000:
                    raise ValueError("PPM3 supports at most 20000 protein residues.")
                number = residue.id[1]
                difference = 1 if previous_number is None else number - previous_number
                if difference < 0:
                    raise ValueError("PPM3 protein residue numbers must not decrease within a chain.")
                sequence_id = number if previous_number is None else sequence_id + max(1, difference)
                if not np.iinfo(np.int32).min <= sequence_id <= np.iinfo(np.int32).max:
                    raise ValueError("PPM3 cannot encode this residue-number range.")
                previous_number = number
                for atom in sorted(heavy_atoms, key=lambda atom: (atom.name.strip().upper() != "N", atom.name)):
                    name = atom.name.strip().upper()
                    if name not in expected:
                        continue
                    coords.append(atom.coord)
                    names.append(name.ljust(4).encode("ascii"))
                    residues.append(native_resname.ljust(4).encode("ascii"))
                    chains.append(chain_id.encode("ascii"))
                    residue_ids.append(sequence_id)
        if not coords:
            raise ValueError("No protein heavy atoms found for native membrane orientation.")
        if len(coords) > 150000:
            raise ValueError("PPM3 supports at most 150000 protein heavy atoms.")
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        if coords.shape != (len(coords), 3) or not np.isfinite(coords).all():
            raise ValueError("PPM3 protein coordinates must contain three finite values per atom.")
        return coords, b"".join(names), b"".join(residues), b"".join(chains), np.array(residue_ids, dtype=np.int32)
