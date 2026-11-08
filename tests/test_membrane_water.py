"""Regression checks for water from the membrane Solvator."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import KDTree

from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.Modeller.Solvator import Solvator, TIP3_def
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.IO.PSFWriter import get_psf_str
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Chain import BaseChain, Ion, Solvent
from crimm.StructEntities.Model import Model
from crimm.StructEntities.Residue import Residue


class MembraneWaterTests(unittest.TestCase):
    def _waters_model(self, oxygen_coords, periodic=False):
        model = Model(0)
        model.topology_loader = TopologyGenerator()
        water = Solvent("WA")
        for number, coord in enumerate(oxygen_coords, start=1):
            residue = Residue(number, "TIP3", "SOLV")
            for name, offset in zip(("OH2", "H1", "H2"), ((0, 0, 0), (0.9572, 0, 0), (-0.24, 0.9266, 0))):
                atom = TIP3_def[name].create_new_atom()
                atom.coord = np.asarray(coord, dtype=float) + offset
                residue.add(atom)
            water.add(residue)
        model.add(water)
        model.topology_loader.generate_solvent(water, "TIP3", QUIET=True)
        if periodic:
            model._solvation_info = {"periodic": True, "box_dims": (20.0, 20.0, 50.0)}
        return model

    def _builder(self, **spec_values):
        model = Model(0)
        protein = BaseChain("A")
        residue = Residue(1, "ALA", "PROA")
        residue.add(Atom("CA", np.array([0.0, 0.0, 23.0]), element="C"))
        protein.add(residue)
        model.add(protein)
        builder = MembraneBuilder(
            model,
            MembraneSpec(
                box_xy=(40.0, 40.0), box_z=70.0,
                upper_leaflet={"POPC": 1}, lower_leaflet={"POPC": 1},
                clash_check=False, **spec_values,
            ),
        )
        builder.prepare_topology()
        builder.add_lipid_bilayer()
        return builder

    def test_generated_waters_stay_outside_membrane(self):
        builder = self._builder()
        self.assertTrue(builder.spec.remove_membrane_core_waters)
        solute = np.array([atom.coord.copy() for atom in builder.model.get_atoms()])
        water_chains = builder.solvate()
        waters = [residue for chain in water_chains for residue in chain]
        self.assertGreater(len(waters), 0)
        oxygen_coords = []
        for chain in water_chains:
            self.assertIs(chain.parent, builder.model)
            self.assertEqual(chain.source, "generated")
            for residue in chain:
                self.assertEqual(residue.resname, "TIP3")
                self.assertEqual(set(residue.child_dict), {"OH2", "H1", "H2"})
                coords = np.array([atom.coord for atom in residue])
                self.assertTrue(np.isfinite(coords).all())
                self.assertTrue((np.abs(coords[:, 2]) >= builder.spec.membrane_core_z).all())
                oxygen_coords.append(residue["OH2"].coord)
        # The Solvator cutoff applies to water oxygens and all solute atoms.
        cell = np.array((*builder.spec.box_xy, builder.spec.box_z))
        distances, _ = KDTree(np.mod(solute, cell), boxsize=cell).query(np.mod(oxygen_coords, cell))
        self.assertTrue((distances >= builder.spec.water_solvcut).all())
        final_solute = np.array([
            atom.coord for chain in builder.model if chain.chain_type != "Solvent"
            for atom in chain.get_atoms()
        ])
        np.testing.assert_array_equal(final_solute, solute)
        self.assertEqual(builder.count_waters(), len(waters))
        self.assertTrue(builder.model._solvation_info["periodic"])
        self.assertTrue(all(residue.topo_definition is not None for residue in waters))
        retained = set(builder.model.get_atoms())
        self.assertTrue(all(atom in retained for chain in water_chains for bond in chain.topology.bonds for atom in bond))

    def test_periodic_solute_extent_does_not_move_existing_atoms(self):
        model = Model(0)
        chain = BaseChain("A")
        residue = Residue(1, "ALA", "A")
        for name, x in (("CA", -15.0), ("CB", 15.0)):
            residue.add(Atom(name, np.array([x, 0.0, 0.0]), element="C"))
        chain.add(residue)
        model.add(chain)
        coords = np.array([atom.coord.copy() for atom in model.get_atoms()])
        with self.assertRaisesRegex(ValueError, "smaller than"):
            Solvator(model).solvate(box_type="ortho", box_dims=(20, 20, 50), orient_coords=False)
        waters = Solvator(model).solvate(
            box_type="ortho", box_dims=(20, 20, 50), orient_coords=False, periodic=True,
        )
        self.assertGreater(sum(len(chain) for chain in waters), 0)
        np.testing.assert_array_equal([atom.coord for atom in chain.get_atoms()], coords)

    def test_periodic_solvent_cutoff_uses_minimum_image(self):
        model = Model(0)
        solvator = Solvator(model)
        solvator.periodic = True
        solvator.box_type = "ortho"
        solvator.box_dims = (20.0, 20.0, 50.0)
        solvator.coords = np.array([[9.8, 0.0, 20.0]])
        solvator.solvcut = 2.8
        waters = np.array([
            [[-9.8, 0, 20], [-8.8, 0, 20], [-10, 1, 20]],
            [[0, 0, 20], [1, 0, 20], [0, 1, 20]],
        ], dtype=float)
        with patch.object(solvator, "create_water_box_coords", return_value=waters):
            np.testing.assert_array_equal(solvator.get_expelled_water_box_coords(), waters[1:])

    def test_periodic_water_fills_faces_without_oxygen_image_overlap(self):
        builder = self._builder()
        waters = [residue for chain in builder.solvate() for residue in chain]
        cell = np.array((*builder.spec.box_xy, builder.spec.box_z))
        oxygen = np.array([residue["OH2"].coord for residue in waters])
        self.assertTrue((cell / 2 - oxygen.max(0) < 1.0).all())
        self.assertTrue((cell / 2 + oxygen.min(0) < 1.0).all())
        self.assertFalse(KDTree(np.mod(oxygen, cell), boxsize=cell).query_pairs(2.4))
        all_coords = np.array([[residue[name].coord for name in ("OH2", "H1", "H2")] for residue in waters])
        self.assertTrue((np.abs(all_coords[:, 1:]) > cell / 2).any())
        source = Solvator(builder.model).water_unit_coords
        source_vectors = source[:, 1:] - source[:, :1]
        for vectors in all_coords[:, 1:] - all_coords[:, :1]:
            self.assertTrue(np.any(np.max(np.abs(source_vectors - vectors), axis=(1, 2)) < 1e-12))

    def test_periodic_seam_cleanup_keeps_whole_water_deterministically(self):
        solvator = Solvator(Model(0))
        solvator.periodic = True
        solvator.box_type = "ortho"
        solvator.box_dims = (20.0, 20.0, 50.0)
        solvator.coords = np.array([[0.0, 0.0, 0.0]])
        solvator.solvcut = 2.8
        waters = np.array([
            [[-9.8, 0, 20], [-8.8, 0, 20], [-10.04, 0.9266, 20]],
            [[9.8, 0, 20], [10.8, 0, 20], [9.56, 0.9266, 20]],
        ])
        with patch.object(solvator, "create_water_box_coords", return_value=waters):
            for _ in range(2):
                np.testing.assert_array_equal(solvator.get_expelled_water_box_coords(), waters[:1])

    def test_periodic_solvation_requires_explicit_fixed_orthorhombic_cell(self):
        for values in (
            {"box_type": "cube", "box_dims": (20, 20, 50), "orient_coords": False},
            {"box_type": "ortho", "box_dims": None, "orient_coords": False},
            {"box_type": "ortho", "box_dims": (20, 20, 50), "orient_coords": True},
            {"box_type": "ortho", "box_dims": (20, -1, 50), "orient_coords": False},
            {"box_type": "ortho", "box_dims": (20, np.nan, 50), "orient_coords": False},
            {"box_type": "ortho", "box_dims": (20, 50), "orient_coords": False},
        ):
            with self.subTest(values=values):
                model = self._waters_model([(0, 0, 20)])
                original_water = model.child_list[0]
                with self.assertRaisesRegex(ValueError, "periodic"):
                    Solvator(model).solvate(periodic=True, **values)
                self.assertIs(model.child_list[0], original_water)

    def test_partial_ion_placement_does_not_replace_waters(self):
        model = self._waters_model([(0, 0, 20), (0.2, 0, 20)])
        waters = list(model.get_residues())
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "place all"):
            Solvator(model).add_ions(concentration=27.75, rng=np.random.default_rng(1))
        self.assertEqual(list(model.get_residues()), waters)
        self.assertFalse(any(chain.chain_type == "Ion" for chain in model))
        self.assertNotIn("n_cation", getattr(model, "_solvation_info", {}))

    def test_fresh_ion_solvator_uses_periodic_distance(self):
        model = self._waters_model([(-9.8, 0, 20), (9.8, 0, 20)], periodic=True)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "place all"):
            Solvator(model).add_ions(concentration=27.75, rng=np.random.default_rng(1))
        self.assertEqual(len(list(model.get_residues())), 2)

    def test_periodic_ion_solute_cutoff_uses_minimum_image(self):
        model = self._waters_model([(-9.8, 0, 20)], periodic=True)
        chain = BaseChain("A")
        residue = Residue(1, "ALA", "A")
        residue.add(Atom("CA", np.array([9.8, 0.0, 20.0]), element="C"))
        chain.add(residue)
        model.add(chain)
        with self.assertWarns(UserWarning):
            placements = Solvator(model)._place_ions_monte_carlo([model["WA"]], ["POT"])
        self.assertEqual(placements, [])

    def test_seeded_ions_have_exact_counts_topology_and_export(self):
        coords = [(x, y, z) for x in (-6, 6) for y in (-6, 6) for z in (-20, 20)]
        outcomes = []
        for _ in range(2):
            model = self._waters_model(coords)
            with patch("crimm.Modeller.Solvator.random.shuffle", side_effect=AssertionError("global RNG used")):
                ion_chain = Solvator(model).add_ions(concentration=6.9375, rng=np.random.default_rng(42))
            self.assertEqual([residue.resname for residue in ion_chain], ["SOD", "CLA"])
            self.assertEqual(model._solvation_info["n_cation"], 1)
            self.assertEqual(model._solvation_info["n_anion"], 1)
            self.assertEqual(len(model["WA"]), 6)
            self.assertEqual(len(model["WA"].topology.bonds), 12)
            self.assertEqual(len(model["WA"].topology.angles), 6)
            retained = set(model.get_atoms())
            self.assertTrue(all(atom in retained for bond in model["WA"].topology.bonds for atom in bond))
            self.assertEqual(model["WA"].total_charge + ion_chain.total_charge, 0)
            self.assertTrue(all(atom.topo_definition is not None for atom in ion_chain.get_atoms()))
            self.assertTrue(all(abs(atom.coord[2]) >= 15 for atom in ion_chain.get_atoms()))
            self.assertIn("!NATOM", get_psf_str(model))
            outcomes.append([atom.coord.copy() for atom in ion_chain.get_atoms()])
        np.testing.assert_array_equal(*outcomes)

    def test_repeat_ion_placement_removes_previous_generated_chain(self):
        model = self._waters_model([(x, y, z) for x in (-6, 6) for y in (-6, 6) for z in (-20, 20)])
        solvator = Solvator(model)
        first = solvator.add_ions(concentration=6.9375, rng=np.random.default_rng(42))
        second = solvator.add_ions(concentration=6.9375, rng=np.random.default_rng(43))
        self.assertIsNone(first.parent)
        self.assertIs(model["IA"], second)
        self.assertEqual(len(model["WA"]), 4)
        self.assertEqual(len(model["WA"].topology.bonds), 8)
        self.assertEqual(len(model["WA"].topology.angles), 4)
        self.assertIn("!NATOM", get_psf_str(model))

    def test_zero_salt_request_clears_previous_ion_count_metadata(self):
        model = self._waters_model([(x, y, z) for x in (-6, 6) for y in (-6, 6) for z in (-20, 20)])
        solvator = Solvator(model)
        solvator.add_ions(concentration=6.9375, rng=np.random.default_rng(42))
        self.assertIsNone(solvator.add_ions(concentration=0))
        self.assertFalse(any(chain.chain_type == "Ion" for chain in model))
        self.assertEqual(model._solvation_info["n_cation"], 0)
        self.assertEqual(model._solvation_info["n_anion"], 0)
        self.assertEqual(model._solvation_info["concentration"], 0)
        self.assertEqual(len(model["WA"]), 6)

    def test_failed_ion_replacement_clears_removed_ion_metadata(self):
        coords = [(x, y, z) for x in (-6, 6) for y in (-6, 6) for z in (-20, 20)]
        model = self._waters_model(coords, periodic=True)
        cell_metadata = dict(model._solvation_info)
        first = Solvator(model).add_ions(concentration=6.9375, rng=np.random.default_rng(42))
        waters = list(model["WA"])
        water_coords = np.array([atom.coord.copy() for atom in model["WA"].get_atoms()])
        water_topology = model["WA"].topology
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "place all"):
            Solvator(model).add_ions(concentration=55.5, rng=np.random.default_rng(43))
        self.assertIsNone(first.parent)
        self.assertFalse(any(chain.chain_type == "Ion" for chain in model))
        self.assertEqual(model._solvation_info, cell_metadata)
        self.assertEqual(list(model["WA"]), waters)
        np.testing.assert_array_equal([atom.coord for atom in model["WA"].get_atoms()], water_coords)
        self.assertIs(model["WA"].topology, water_topology)
        self.assertEqual(len(water_topology.bonds), 2 * len(waters))
        self.assertEqual(len(water_topology.angles), len(waters))

    def test_preserved_ions_obey_periodic_ion_distance(self):
        model = self._waters_model([(-9.8, 0, 20)], periodic=True)
        ions = Ion("IS")
        residue = model.topology_loader.cur_defs["POT"].create_residue(resseq=1)
        residue.atoms[0].coord = np.array([9.8, 0.0, 20.0])
        ions.add(residue)
        model.add(ions)
        with self.assertWarns(UserWarning):
            placements = Solvator(model)._place_ions_monte_carlo([model["WA"]], ["CLA"])
        self.assertEqual(placements, [])

    def test_unsupported_ion_conditions_fail_before_water_mutation(self):
        for values in ({"concentration": -1}, {"concentration": np.nan}, {"cation": "CAL"}, {"anion": "OH"}):
            with self.subTest(values=values):
                model = self._waters_model([(0, 0, 20), (8, 0, 20)])
                waters = list(model.get_residues())
                with self.assertRaises(ValueError):
                    Solvator(model).add_ions(**values)
                self.assertEqual(list(model.get_residues()), waters)

    def test_fixed_calcium_uses_selected_waters_and_survives_salt_placement(self):
        model = self._waters_model([(x, y, z) for x in (-6, 6) for y in (-6, 6) for z in (-20, 20)])
        upper_waters = [residue for residue in model["WA"] if residue["OH2"].coord[2] > 0]
        calcium = Solvator(model).add_ions_by_count(
            {"CAL": 1}, eligible_waters=upper_waters, rng=np.random.default_rng(42))
        self.assertEqual(calcium.source, "membrane_counterions")
        self.assertEqual(calcium.total_charge, 2)
        self.assertGreater(next(calcium.get_atoms()).coord[2], 0)
        anions = Solvator(model).add_ions(concentration=0, rng=np.random.default_rng(43))
        self.assertIs(calcium.parent, model)
        self.assertEqual([residue.resname for residue in anions], ["CLA", "CLA"])
        self.assertEqual(sum(chain.total_charge for chain in model), 0)
        self.assertEqual(model._solvation_info["fixed_ion_counts"], {"CAL": 1})
        self.assertEqual(len(model["WA"]), 5)
        self.assertIn("!NATOM", get_psf_str(model))
        self.assertIn("3 ions (CAL,CLA)", get_psf_str(model))

    def test_partial_fixed_counterion_placement_leaves_waters_unchanged(self):
        model = self._waters_model([(0, 0, 20), (0, 0, -20)])
        waters = list(model["WA"])
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "fixed counterions"):
            Solvator(model).add_ions_by_count({"CAL": 2}, eligible_waters=waters[:1])
        self.assertEqual(list(model["WA"]), waters)
        self.assertFalse(any(chain.chain_type == "Ion" for chain in model))

    def test_invalid_fixed_counterion_counts_fail_before_water_mutation(self):
        for counts in ({"CAL": -1}, {"CAL": 0.5}, {"CAL": np.nan}, {"UNDEFINED": 1}):
            model = self._waters_model([(0, 0, 20)])
            water = model["WA"][1]
            with self.assertRaises(ValueError):
                Solvator(model).add_ions_by_count(counts)
            self.assertIs(model["WA"][1], water)

    def test_membrane_ions_use_retained_complete_water_and_actual_counts(self):
        builder = self._builder()
        builder.model.detach_child("A")
        solute = np.array([atom.coord.copy() for atom in builder.model.get_atoms()])
        builder.solvate()
        initial_water_count = builder.count_waters()
        builder.add_ions()
        ions = [residue for chain in builder.model if chain.chain_type == "Ion" for residue in chain]
        self.assertGreater(len(ions), 0)
        self.assertEqual(builder.count_waters(), initial_water_count - len(ions))
        metadata = builder.model._solvation_info
        self.assertEqual(metadata["n_cation"], sum(residue.resname == "POT" for residue in ions))
        self.assertEqual(metadata["n_anion"], sum(residue.resname == "CLA" for residue in ions))
        self.assertEqual(sum(chain.total_charge for chain in builder.model), 0)
        ion_coords = []
        waters = []
        for chain in builder.model:
            for residue in chain:
                self.assertIsNotNone(residue.topo_definition)
                self.assertTrue(all(atom.topo_definition is not None for atom in residue))
                if chain.chain_type == "Ion":
                    ion_coords.append(residue.atoms[0].coord)
                    self.assertGreaterEqual(abs(residue.atoms[0].coord[2]), builder.spec.membrane_core_z)
                elif chain.chain_type == "Solvent":
                    waters.append(residue)
                    self.assertEqual(set(residue.child_dict), {"OH2", "H1", "H2"})
                    self.assertTrue(all(abs(atom.coord[2]) >= builder.spec.membrane_core_z for atom in residue))
        self.assertEqual(len(waters), builder.count_waters())
        cell = np.array((*builder.spec.box_xy, builder.spec.box_z))
        distances, _ = KDTree(np.mod(solute, cell), boxsize=cell).query(np.mod(ion_coords, cell))
        self.assertTrue((distances >= 5.0).all())
        self.assertFalse(KDTree(np.mod(ion_coords, cell), boxsize=cell).query_pairs(5.0))
        for chain in builder.model:
            if chain.chain_type == "Solvent":
                self.assertEqual(len(chain.topology.bonds), 2 * len(chain))
                self.assertEqual(len(chain.topology.angles), len(chain))
        self.assertIn("!NATOM", get_psf_str(builder.model))

    def test_core_cleanup_can_be_disabled(self):
        builder = self._builder(remove_membrane_core_waters=False)
        water_chains = builder.solvate()
        oxygen_z = np.array([
            residue["OH2"].coord[2] for chain in water_chains for residue in chain
        ])
        self.assertTrue((np.abs(oxygen_z) < builder.spec.membrane_core_z).any())

    def test_empty_water_chains_are_not_returned(self):
        builder = self._builder(membrane_core_z=100.0)
        self.assertEqual(builder.solvate(), [])
        self.assertEqual(builder.count_waters(), 0)
        self.assertFalse(any(chain.chain_type == "Solvent" for chain in builder.model))

    def test_cleanup_removes_whole_water_if_a_hydrogen_enters_core(self):
        model = Model(0)
        chain = Solvent("WA")
        for number, z_values in ((1, (15.3, 14.8, 15.3)), (2, (16.2, 16.2, 16.2))):
            residue = Residue(number, "TIP3", "SOLV")
            for name, z in zip(("OH2", "H1", "H2"), z_values):
                residue.add(Atom(name, np.array([0.0, 0.0, z]), element="O" if name == "OH2" else "H"))
            chain.add(residue)
        model.add(chain)
        builder = MembraneBuilder(
            model,
            MembraneSpec(
                box_xy=(40.0, 40.0), box_z=70.0,
                upper_leaflet={}, lower_leaflet={}, use_lipid_library=False,
            ),
        )
        self.assertEqual(builder._remove_waters_in_membrane_core(), 1)
        self.assertEqual([residue.id[1] for residue in chain], [2])
        self.assertEqual(set(chain[2].child_dict), {"OH2", "H1", "H2"})


if __name__ == "__main__":
    unittest.main()
