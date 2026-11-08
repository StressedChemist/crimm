"""Regression checks for membrane placement."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import KDTree

from crimm.Modeller.MembraneBuilder import (
    MembraneBuilder,
    MembraneSpec,
    PackedHeadPosition,
)
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Chain import BaseChain, Lipid
from crimm.StructEntities.Model import Model
from crimm.StructEntities.Residue import Residue
from crimm.Modeller.TopoLoader import TopologyGenerator


class MembraneGeometryTests(unittest.TestCase):
    def test_membrane_contact_tree_is_reused_until_placed_molecules_change(self):
        builder = MembraneBuilder(Model(0), MembraneSpec(
            box_xy=(40, 40), box_z=80, upper_leaflet={}, lower_leaflet={}, use_lipid_library=False))
        candidate = Residue(1, "POPC", "MEMB")
        candidate.add(Atom("C1", np.array([0., 0., 20.]), element="C"))
        builder._placed_membrane_coords_cache = [np.array([[20., 0., 20.]])]
        with patch("crimm.Modeller.MembraneBuilder.KDTree", wraps=KDTree) as tree:
            self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
            self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
            self.assertEqual(tree.call_count, 1)
            builder._placed_membrane_coords_cache.append(np.array([[0., 0., 20.]]))
            self.assertTrue(builder._has_bad_contacts(candidate, "Lipid"))
            self.assertEqual(tree.call_count, 2)
            builder._placed_membrane_coords_cache = [np.array([[20., 0., 20.]])]
            self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
            self.assertEqual(tree.call_count, 3)

    def test_existing_membrane_coordinates_do_not_use_a_stale_contact_tree(self):
        model = Model(0)
        membrane = Lipid("MEMB")
        existing = Residue(1, "POPC", "MEMB")
        existing.add(Atom("C1", np.array([20., 0., 20.]), element="C"))
        membrane.add(existing)
        model.add(membrane)
        builder = MembraneBuilder(model, MembraneSpec(box_xy=(40, 40), box_z=80,
            upper_leaflet={}, lower_leaflet={}, use_lipid_library=False))
        builder._placed_membrane_coords_cache = []
        candidate = Residue(2, "POPC", "MEMB")
        candidate.add(Atom("C1", np.array([0., 0., 20.]), element="C"))
        self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
        existing["C1"].coord = candidate["C1"].coord.copy()
        self.assertTrue(builder._has_bad_contacts(candidate, "Lipid"))

    @staticmethod
    def _chirality(residue):
        return np.linalg.det([
            residue[name].coord - residue["C2"].coord
            for name in ("C1", "C3", "O21")
        ])

    def test_leaflet_flip_preserves_conformer_geometry_and_chirality(self):
        builder = MembraneBuilder(
            Model(0), MembraneSpec(box_xy=(40., 40.), box_z=80.,
                                  upper_leaflet={}, lower_leaflet={}),
        )
        builder.prepare_topology()
        definitions, _ = builder.topology.load_residue_definitions("Lipid")
        residue = definitions["POPC"].create_residue(resseq=1)
        conformer = builder.lipid_library.random_conformer("POPC", builder.rng)
        builder._apply_conformer_coords(residue, conformer)
        before = np.array([atom.coord.copy() for atom in residue])
        chirality = self._chirality(residue)
        builder._flip_lipid_z(residue)
        after = np.array([atom.coord for atom in residue])
        self.assertAlmostEqual(self._chirality(residue), chirality, places=12)
        np.testing.assert_allclose(
            np.linalg.norm(before[:, None] - before[None, :], axis=2),
            np.linalg.norm(after[:, None] - after[None, :], axis=2), atol=1e-12,
        )

    def test_periodic_packing_preserves_heads_and_chirality(self):
        builder = MembraneBuilder(
            Model(0), MembraneSpec(box_xy=(32., 32.), box_z=80.,
                                  upper_leaflet={"POPC": 16}, lower_leaflet={"POPC": 16}),
        )
        builder.prepare_topology()
        builder.add_lipid_bilayer()
        residues = list(builder.model["MEMB"])
        self.assertEqual(len(residues), 32)
        self.assertTrue(all(self._chirality(residue) < 0 for residue in residues))
        np.testing.assert_allclose([r["P"].coord[2] for r in residues], [20.] * 16 + [-20.] * 16, atol=1e-12)
        atoms = [atom for residue in residues for atom in residue if atom.element != "H"]
        cell = np.array([32., 32., 80.])
        pairs = KDTree(np.array([atom.coord for atom in atoms]) % cell, boxsize=cell).query_pairs(1.2)
        self.assertFalse(any(atoms[i].parent is not atoms[j].parent for i, j in pairs))

    def test_retry_rotation_preserves_head_and_conformer_geometry(self):
        builder = MembraneBuilder(
            Model(0), MembraneSpec(box_xy=(40., 40.), box_z=80.,
                                  upper_leaflet={"POPC": 1, "CHL1": 1}, lower_leaflet={}),
        )
        builder.prepare_topology()
        for resname, chain_type in (("POPC", "Lipid"), ("CHL1", "Sterol")):
            with self.subTest(resname=resname):
                residue = builder._create_residue(resname, chain_type, 1, "MEMB", np.array([7., 8., 20.]), 1)
                head = np.mean([atom.coord for atom in builder._head_atoms(residue, chain_type)], axis=0)
                before = np.array([atom.coord.copy() for atom in residue])
                chirality = self._chirality(residue) if resname == "POPC" else None
                builder._rotate_residue_z_about_head(residue, chain_type, np.pi / 3)
                after = np.array([atom.coord for atom in residue])
                np.testing.assert_allclose(np.mean([atom.coord for atom in builder._head_atoms(residue, chain_type)], axis=0), head, atol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(before[:, None] - before[None, :], axis=2), np.linalg.norm(after[:, None] - after[None, :], axis=2), atol=1e-12)
                if chirality is not None:
                    self.assertAlmostEqual(self._chirality(residue), chirality, places=12)

    def test_loader_sharing_preserves_existing_definitions(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                model = Model(0)
                loader = TopologyGenerator()
                protein_definitions, _ = loader.load_residue_definitions("Polypeptide(L)")
                if existing:
                    model.topology_loader = loader
                builder = MembraneBuilder(
                    model, MembraneSpec(box_xy=(32., 32.), box_z=80.,
                                        upper_leaflet={}, lower_leaflet={}, use_lipid_library=False),
                )
                builder.prepare_topology()
                self.assertIs(model.topology_loader, builder.topology)
                if existing:
                    self.assertIs(builder.topology, loader)
                    self.assertIs(builder.topology.res_def_dict["protein"], protein_definitions)

    def test_contacts_include_periodic_images(self):
        model = Model(0)
        chain = BaseChain("A")
        protein = Residue(1, "ALA", "PROA")
        protein.add(Atom("CA", np.array([-15.8, 0., 20.]), element="C"))
        chain.add(protein)
        model.add(chain)
        builder = MembraneBuilder(
            model, MembraneSpec(box_xy=(32., 32.), box_z=80., upper_leaflet={},
                                lower_leaflet={}, use_lipid_library=False),
        )
        candidate = Residue(1, "POPC", "MEMB")
        candidate.add(Atom("C1", np.array([15.8, 0., 20.]), element="C"))
        self.assertTrue(builder._has_bad_contacts(candidate, "Lipid"))
        candidate["C1"].coord = np.array([0., 0., 20.])
        self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
        model.detach_child("A")
        builder._placed_membrane_coords_cache = [np.array([[-15.8, 0., 20.]])]
        candidate["C1"].coord = np.array([15.8, 0., 20.])
        self.assertTrue(builder._has_bad_contacts(candidate, "Lipid"))
        builder._placed_membrane_coords_cache = [np.array([[-1e-15, 0., 20.]])]
        candidate["C1"].coord = np.array([0., 0., 20.])
        self.assertTrue(builder._has_bad_contacts(candidate, "Lipid"))

    def test_exhausted_packing_does_not_return_a_clash(self):
        builder = MembraneBuilder(
            Model(0), MembraneSpec(box_xy=(32., 32.), box_z=80., upper_leaflet={},
                                  lower_leaflet={}, max_repack_attempts=0),
        )
        builder.prepare_topology()
        residue = builder._create_residue("POPC", "Lipid", 1, "MEMB", np.array([0., 0., 20.]), 1)
        builder._placed_membrane_coords_cache = [builder._residue_heavy_atom_coords(residue)]
        with self.assertRaisesRegex(ValueError, "without periodic clashes"):
            builder._replace_with_nonclashing_residue("POPC", "Lipid", 1, "MEMB", np.array([0., 0., 20.]), 1, residue)

    def test_local_repacking_can_clear_a_two_angstrom_obstacle(self):
        model = Model(0)
        chain = BaseChain("A")
        protein = Residue(1, "ALA", "PROA")
        protein.add(Atom("CA", np.array([0., 0., 20.]), element="C"))
        chain.add(protein)
        model.add(chain)
        builder = MembraneBuilder(model, MembraneSpec(
            box_xy=(32., 32.), box_z=80., upper_leaflet={}, lower_leaflet={},
            use_lipid_library=False, max_repack_attempts=1,
        ))
        candidate = Residue(1, "POPC", "MEMB")
        candidate.add(Atom("P", np.array([0., 0., 20.]), element="P"))
        builder._repack_residue_if_needed(candidate, "Lipid")
        self.assertFalse(builder._has_bad_contacts(candidate, "Lipid"))
        self.assertEqual(candidate["P"].coord[2], 20.)

    def test_wider_solvation_xy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match box_xy"):
            MembraneBuilder(
                Model(0), MembraneSpec(box_xy=(32., 32.), box_z=80., solvation_box=(40., 40., 80.),
                                      upper_leaflet={}, lower_leaflet={}, use_lipid_library=False),
            )

    def test_lipid_z_in_grid_and_packed_placement(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                builder = MembraneBuilder(
                    Model(0),
                    MembraneSpec(
                        box_xy=(40.0, 40.0), box_z=80.0,
                        upper_leaflet={"POPC": 1}, lower_leaflet={"POPC": 1},
                        lipid_z=27.5, clash_check=False,
                    ),
                )
                if packed:
                    builder.packed_head_positions = [
                        PackedHeadPosition("POPC", np.array([1.0, 2.0, 3.0]), 1),
                        PackedHeadPosition("POPC", np.array([-1.0, -2.0, -3.0]), -1),
                    ]
                builder.prepare_topology()
                builder.add_lipid_bilayer()
                heads = [residue["P"].coord[2] for residue in builder.model["MEMB"]]
                np.testing.assert_allclose(heads, [27.5, -27.5], atol=1e-12)

    def test_empty_leaflets(self):
        for upper, lower in (
            ({}, {}),
            ({"POPC": 0}, {"POPC": 0}),
            ({"POPC": 1}, {}),
            ({}, {"POPC": 1}),
        ):
            with self.subTest(upper=upper, lower=lower):
                builder = MembraneBuilder(
                    Model(0),
                    MembraneSpec(
                        box_xy=(40.0, 40.0), box_z=80.0,
                        upper_leaflet=upper, lower_leaflet=lower,
                        use_lipid_library=False,
                    ),
                )
                builder.prepare_topology()
                builder.add_lipid_bilayer()
                expected = sum(upper.values()) + sum(lower.values())
                self.assertEqual(builder.count_lipids(), expected)
                self.assertEqual(len(builder.model), int(expected > 0))

    def test_empty_packed_heads_preserve_protein(self):
        model = Model(0)
        chain = BaseChain("A")
        residue = Residue(1, "ALA", "PROA")
        residue.add(Atom("CA", np.array([2.0, 3.0, 4.0]), element="C"))
        chain.add(residue)
        model.add(chain)
        builder = MembraneBuilder(
            model,
            MembraneSpec(
                box_xy=(40.0, 40.0), box_z=80.0,
                upper_leaflet={}, lower_leaflet={}, use_lipid_library=False,
            ),
        )
        builder._align_packed_heads_to_protein_xy([])
        builder.packed_head_positions = []
        builder.add_lipid_bilayer()
        self.assertEqual(len(model), 1)
        np.testing.assert_array_equal(residue["CA"].coord, [2.0, 3.0, 4.0])

    def test_negative_leaflet_count_fails(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MembraneBuilder(
                Model(0),
                MembraneSpec(
                    box_xy=(40.0, 40.0), box_z=80.0,
                    upper_leaflet={"POPC": -1}, lower_leaflet={},
                    use_lipid_library=False,
                ),
            )


if __name__ == "__main__":
    unittest.main()
