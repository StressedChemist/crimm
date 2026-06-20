"""Orient proteins for membrane building.

This module implements the geometry used by the PERMM/PPM orientation step:
rotate a coordinate set by angle ``teta`` around an axis in the XY plane whose 
direction is defined by ``phi``.  The default scorer is an atom-level
PERMM-style transfer-energy approximation: each heavy atom is assigned a
depth-dependent cost or reward for being in the membrane core/interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


HYDROPHOBIC_RESIDUES = frozenset(
    {
        "ALA",
        "CYS",
        "ILE",
        "LEU",
        "MET",
        "PHE",
        "PRO",
        "VAL",
        "TRP",
        "TYR",
    }
)

POLAR_RESIDUES = frozenset(
    {
        "ASN",
        "GLN",
        "SER",
        "THR",
        "HSD",
        "HSE",
        "HSP",
        "HIS",
    }
)

CHARGED_RESIDUES = frozenset(
    {
        "ARG",
        "LYS",
        "ASP",
        "GLU",
    }
)

PROTEIN_CHAIN_TYPES = frozenset({"Polypeptide(L)", "Protein", "Chain"})

BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT", "H", "HA"})
POSITIVE_RESIDUES = frozenset({"ARG", "LYS", "HIS", "HSD", "HSE", "HSP"})
NEGATIVE_RESIDUES = frozenset({"ASP", "GLU"})


@dataclass(frozen=True)
class MembraneOrientation:
    """Rigid-body transform selected by the membrane orientation search."""

    phi: float
    teta: float
    shift_z: float
    score: float


class MembraneOrienter:
    """Orient a protein relative to an implicit membrane centered on XY."""

    def __init__(
        self,
        model,
        membrane_core_z: float = 15.0,
        interface_z: float = 20.0,
        interface_width: float = 3.0,
        phi_step: float = 10.0,
        teta_step: float = 10.0,
        shift_step: float = 2.0,
        shift_limit: float = 20.0,
        score_mode: str = "atom_transfer",
        use_exposure_weights: bool = True,
        exposure_cutoff: float = 8.0,
        exposure_neighbor_scale: float = 14.0,
        min_exposure_weight: float = 0.15,
        refine: bool = True,
        refine_factor: int = 5,
    ):
        self.model = model
        self.membrane_core_z = float(membrane_core_z)
        self.interface_z = float(interface_z)
        self.interface_width = float(interface_width)
        self.phi_step = float(phi_step)
        self.teta_step = float(teta_step)
        self.shift_step = float(shift_step)
        self.shift_limit = float(shift_limit)
        self.score_mode = score_mode
        self.use_exposure_weights = bool(use_exposure_weights)
        self.exposure_cutoff = float(exposure_cutoff)
        self.exposure_neighbor_scale = float(exposure_neighbor_scale)
        self.min_exposure_weight = float(min_exposure_weight)
        self.refine = bool(refine)
        self.refine_factor = int(refine_factor)

    def orient(self, in_place: bool = True) -> MembraneOrientation:
        """Find and optionally apply a membrane orientation transform."""

        coords = self._protein_heavy_atom_coords()
        if len(coords) == 0:
            raise ValueError("No protein heavy atoms found for membrane orientation.")

        center = coords.mean(axis=0)
        if self.score_mode == "atom_transfer":
            search_points = self._atom_transfer_points(center=center)
        elif self.score_mode == "residue":
            search_points = self._residue_points(center=center)
        else:
            raise ValueError(f"Unknown membrane orientation score mode: {self.score_mode}")

        if not search_points:
            raise ValueError("No protein points found for membrane orientation.")

        orientation = self.search_orientation(search_points)

        if in_place:
            self.apply_orientation(orientation, center=center)

        return orientation

    def search_orientation(self, residue_points) -> MembraneOrientation:
        """Grid-search PERMM-style phi/teta plus a Z shift."""

        phi_values = np.arange(0.0, 180.0, self.phi_step)
        teta_values = np.arange(-90.0, 90.0 + self.teta_step, self.teta_step)
        shift_values = np.arange(
            -self.shift_limit,
            self.shift_limit + self.shift_step,
            self.shift_step,
        )

        best = self._search_orientation_grid(
            residue_points,
            phi_values,
            teta_values,
            shift_values,
        )

        if not self.refine or self.refine_factor <= 1:
            return best

        phi_step = self.phi_step / self.refine_factor
        teta_step = self.teta_step / self.refine_factor
        shift_step = self.shift_step / self.refine_factor

        phi_values = self._wrapped_phi_values(
            best.phi - self.phi_step,
            best.phi + self.phi_step,
            phi_step,
        )
        teta_values = np.arange(
            max(-90.0, best.teta - self.teta_step),
            min(90.0, best.teta + self.teta_step) + teta_step,
            teta_step,
        )
        shift_values = np.arange(
            max(-self.shift_limit, best.shift_z - self.shift_step),
            min(self.shift_limit, best.shift_z + self.shift_step) + shift_step,
            shift_step,
        )

        return self._search_orientation_grid(
            residue_points,
            phi_values,
            teta_values,
            shift_values,
            best=best,
        )

    def _search_orientation_grid(
        self,
        residue_points,
        phi_values: np.ndarray,
        teta_values: np.ndarray,
        shift_values: np.ndarray,
        best: MembraneOrientation | None = None,
    ) -> MembraneOrientation:
        """Evaluate a grid of PERMM-style orientation variables."""

        if best is None:
            best = MembraneOrientation(phi=0.0, teta=0.0, shift_z=0.0, score=np.inf)

        coords = np.array([point[0] for point in residue_points], dtype=float)

        for phi in phi_values:
            for teta in teta_values:
                rotated = coords @ self.permm_rotation_matrix(phi, teta).T
                for shift_z in shift_values:
                    shifted_z = rotated[:, 2] + shift_z
                    score = self._score_depths(residue_points, shifted_z)
                    if score < best.score:
                        best = MembraneOrientation(
                            phi=float(phi),
                            teta=float(teta),
                            shift_z=float(shift_z),
                            score=float(score),
                        )

        return best

    @staticmethod
    def _wrapped_phi_values(start: float, stop: float, step: float) -> np.ndarray:
        """Return phi values wrapped into the PERMM 0-180 degree interval."""

        values = np.arange(start, stop + step, step)
        return np.unique(np.mod(values, 180.0))

    def apply_orientation(
        self,
        orientation: MembraneOrientation,
        center: np.ndarray | None = None,
    ):
        """Apply an orientation transform to the model in place."""

        if center is None:
            center = self._protein_heavy_atom_coords().mean(axis=0)

        rotation = self.permm_rotation_matrix(orientation.phi, orientation.teta)
        shift = np.array([0.0, 0.0, orientation.shift_z], dtype=float)

        for atom in self._protein_atoms(include_hydrogen=True):
            atom.coord = ((atom.coord - center) @ rotation.T) + shift

    @staticmethod
    def permm_rotation_matrix(phi: float, teta: float) -> np.ndarray:
        """Return the rotation matrix from PERMM ``tilting.f``.

        ``phi`` and ``teta`` are in degrees.  ``teta`` is rotation around an
        axis in the XY plane.  ``phi`` defines that axis direction.
        """

        phi_rad = np.deg2rad(phi)
        teta_rad = np.deg2rad(teta)

        a = np.sin(phi_rad)
        b = np.cos(phi_rad)
        si = np.sin(teta_rad)
        co = np.cos(teta_rad)
        co1 = 1.0 - co

        return np.array(
            [
                [co + a * a * co1, a * b * co1, b * si],
                [a * b * co1, co + b * b * co1, -a * si],
                [-b * si, a * si, co],
            ],
            dtype=float,
        )

    def _score_depths(self, points, z_values: np.ndarray) -> float:
        if self.score_mode == "atom_transfer":
            return self._score_atom_depths(points, z_values)

        resnames = [point[1] for point in points]
        return self._score_residue_depths(resnames, z_values)

    def _score_atom_depths(self, atom_points, z_values: np.ndarray) -> float:
        """Score atom transfer energy in a symmetric membrane slab.

        This is a CRIMM-native approximation of the PERMM idea, not a direct
        line-by-line port.  Hydrophobic exposed atoms are rewarded in the core;
        polar/charged exposed atoms are penalized in the core and weakly
        rewarded near the interface.
        """

        abs_z = np.abs(z_values)
        core = self._core_profile(abs_z)
        interface = self._interface_profile(abs_z)

        score = 0.0
        total_weight = 0.0

        for i, point in enumerate(atom_points):
            _, core_energy, interface_energy, weight = point
            score += weight * (core_energy * core[i] + interface_energy * interface[i])
            total_weight += weight

        return score / max(total_weight, 1.0)

    def _score_residue_depths(self, resnames: list[str], z_values: np.ndarray) -> float:
        """Score residue depths relative to a symmetric membrane slab."""

        score = 0.0
        abs_z = np.abs(z_values)

        for resname, depth in zip(resnames, abs_z):
            if resname in HYDROPHOBIC_RESIDUES:
                score += self._quadratic_penalty(depth, target=0.0, width=self.membrane_core_z)
            elif resname in CHARGED_RESIDUES:
                score += 3.0 * self._inside_penalty(depth, cutoff=self.membrane_core_z)
            elif resname in POLAR_RESIDUES:
                score += self._inside_penalty(depth, cutoff=self.membrane_core_z)
            else:
                score += 0.25 * self._inside_penalty(depth, cutoff=self.membrane_core_z)

        return score / max(len(resnames), 1)

    def _core_profile(self, abs_z: np.ndarray) -> np.ndarray:
        """Smooth membrane-core occupancy profile."""

        softness = 1.5
        return 1.0 / (1.0 + np.exp((abs_z - self.membrane_core_z) / softness))

    def _interface_profile(self, abs_z: np.ndarray) -> np.ndarray:
        """Smooth headgroup/interface occupancy profile."""

        return np.exp(
            -0.5 * ((abs_z - self.interface_z) / self.interface_width) ** 2
        )

    @staticmethod
    def _quadratic_penalty(value: float, target: float, width: float) -> float:
        distance = max(0.0, abs(value - target) - width)
        return distance * distance

    @staticmethod
    def _inside_penalty(value: float, cutoff: float) -> float:
        distance = max(0.0, cutoff - value)
        return distance * distance

    def _residue_points(self, center: np.ndarray):
        """Return residue center points translated relative to protein center."""

        points = []
        for chain in self._protein_chains():
            for residue in chain:
                atoms = [
                    atom
                    for atom in residue.get_atoms()
                    if not self._is_hydrogen(atom)
                ]
                if not atoms:
                    continue

                coords = np.array([atom.coord for atom in atoms], dtype=float)
                points.append((coords.mean(axis=0) - center, residue.resname.strip().upper()))

        return points

    def _atom_transfer_points(self, center: np.ndarray):
        """Return atom points plus transfer-energy parameters."""

        coords = []
        terms = []

        for chain in self._protein_chains():
            for residue in chain:
                resname = residue.resname.strip().upper()
                for atom in residue.get_atoms():
                    if self._is_hydrogen(atom):
                        continue

                    atom_name = atom.get_name().strip().upper()
                    core_energy, interface_energy, weight = self._atom_transfer_terms(
                        resname,
                        atom_name,
                        getattr(atom, "element", "").strip().upper(),
                    )
                    coords.append(atom.coord.copy())
                    terms.append((core_energy, interface_energy, weight))

        if not coords:
            return []

        coords = np.array(coords, dtype=float)
        exposure_weights = np.ones(len(coords), dtype=float)
        if self.use_exposure_weights:
            exposure_weights = self._approximate_exposure_weights(coords)

        points = []
        for coord, term, exposure_weight in zip(coords, terms, exposure_weights):
            core_energy, interface_energy, weight = term
            points.append(
                (
                    coord - center,
                    core_energy,
                    interface_energy,
                    weight * exposure_weight,
                )
            )

        return points

    def _approximate_exposure_weights(self, coords: np.ndarray) -> np.ndarray:
        """Estimate atom exposure from local heavy-atom neighbor density.

        PERMM uses solvent-accessible surface area.  This inexpensive proxy
        downweights atoms with many nearby heavy atoms and preserves exposed
        atoms as the main drivers of orientation.
        """

        neighbor_counts = np.zeros(len(coords), dtype=float)
        cutoff2 = self.exposure_cutoff * self.exposure_cutoff
        chunk_size = 512

        for start in range(0, len(coords), chunk_size):
            stop = min(start + chunk_size, len(coords))
            delta = coords[start:stop, None, :] - coords[None, :, :]
            dist2 = np.sum(delta * delta, axis=2)
            counts = np.sum(dist2 < cutoff2, axis=1) - 1
            neighbor_counts[start:stop] = counts

        exposure = np.exp(-neighbor_counts / self.exposure_neighbor_scale)
        return np.clip(exposure, self.min_exposure_weight, 1.0)

    def _atom_transfer_terms(
        self,
        resname: str,
        atom_name: str,
        element: str,
    ) -> tuple[float, float, float]:
        """Return core energy, interface energy, and exposure weight."""

        sidechain_weight = 1.0
        backbone_weight = 0.35
        weight = backbone_weight if atom_name in BACKBONE_ATOMS else sidechain_weight

        if self._is_charged_sidechain_atom(resname, atom_name):
            return 9.0, -1.0, weight

        if element in {"O", "N"}:
            if atom_name in BACKBONE_ATOMS:
                return 1.2, -0.15, weight
            return 3.5, -0.6, weight

        if element == "S":
            if resname in HYDROPHOBIC_RESIDUES:
                return -0.8, 0.0, weight
            return 1.0, -0.2, weight

        if element == "C":
            if atom_name in BACKBONE_ATOMS:
                return 0.1, 0.0, weight
            if resname in HYDROPHOBIC_RESIDUES:
                return -1.1, 0.1, weight
            if resname in CHARGED_RESIDUES:
                return 0.5, -0.1, weight
            return -0.25, 0.0, weight

        return 0.2, 0.0, weight

    @staticmethod
    def _is_charged_sidechain_atom(resname: str, atom_name: str) -> bool:
        if resname in NEGATIVE_RESIDUES and atom_name.startswith(("OD", "OE")):
            return True
        if resname == "LYS" and atom_name == "NZ":
            return True
        if resname == "ARG" and atom_name.startswith(("NE", "NH")):
            return True
        if resname in {"HIS", "HSD", "HSE", "HSP"} and atom_name.startswith(("ND", "NE")):
            return True
        return False

    def _protein_heavy_atom_coords(self) -> np.ndarray:
        return np.array(
            [
                atom.coord
                for atom in self._protein_atoms(include_hydrogen=False)
            ],
            dtype=float,
        )

    def _protein_atoms(self, include_hydrogen: bool = False) -> Iterable:
        for chain in self._protein_chains():
            for atom in chain.get_atoms():
                if not include_hydrogen and self._is_hydrogen(atom):
                    continue
                yield atom

    def _protein_chains(self):
        for chain in self.model:
            chain_type = getattr(chain, "chain_type", None)
            if chain_type in {"Lipid", "Sterol", "Solvent", "Ion"}:
                continue
            if chain_type is None or chain_type in PROTEIN_CHAIN_TYPES:
                yield chain

    @staticmethod
    def _is_hydrogen(atom) -> bool:
        element = getattr(atom, "element", "").strip().upper()
        name = atom.get_name().strip().upper()
        return element == "H" or name.startswith("H")
