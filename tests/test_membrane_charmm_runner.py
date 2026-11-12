"""Input corruption checks for the CHARMM membrane runner, without a native runtime."""

import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from crimm.IO import write_crd, write_psf
from crimm.Modeller import TopologyGenerator
from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Chain import Solvent
from crimm.StructEntities.Model import Model
from crimm.StructEntities.Residue import Residue
from scripts.membrane_pycharmm import charmm_worker, coordinate_pair, digest, geometry_checks, main, nonbonded_script, relaxation_script


class MembraneCharmmRunnerTests(unittest.TestCase):
    def test_dynamics_trials_receive_the_one_hour_native_deadline(self):
        cases = (([], 900), (["--nvt-steps", "100"], 3600),
                 (["--npt-steps", "100"], 3600), (["--membrane", "aPM"], 3600))
        for flags, expected in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "new-run"
                args = ["runner", "--output", str(output), "--charmm-lib", "/missing", *flags]
                with patch("sys.argv", args), \
                     patch("scripts.membrane_pycharmm.digest", return_value="0" * 64), \
                     patch("scripts.membrane_pycharmm.prepare", return_value=(None, None)), \
                     patch("scripts.membrane_pycharmm.subprocess.run",
                           side_effect=subprocess.TimeoutExpired("native worker", expected)) as run:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        main()
                self.assertEqual(run.call_args.kwargs["timeout"], expected)
                self.assertEqual(json.loads((output / "result.json").read_text())["charmm_timeout_seconds"], expected)

    def test_short_range_warmup_and_full_pme_settings_remain_separate(self):
        full = nonbonded_script((83.04, 83.04, 100))
        short = nonbonded_script((83.04, 83.04, 100), short_range=True)
        self.assertIn("ctonnb 10 ctofnb 12 cutnb 16 cutim 16", full)
        self.assertIn("ewald pmew", full)
        self.assertIn("ctonnb 5 ctofnb 6 cutnb 8 cutim 8", short)
        self.assertIn("noewald nopme", short)
        self.assertNotIn("ewald pmew", short)
        self.assertEqual(full, nonbonded_script((83.04, 83.04, 100)))
        self.assertLess(max(map(len, (full + short).splitlines())), 200)

    def test_nvt_trial_uses_langevin_and_writes_restart_and_trajectory(self):
        script = relaxation_script("nvt", 1000, 310.15, 12345, 2000)
        self.assertIn("dyna leap langevin start", script)
        self.assertIn("nstep 1000 timestep 0.001", script)
        self.assertIn("nvt.rst", script)
        self.assertIn("nvt.dcd", script)
        self.assertNotIn("pcons", script)
        self.assertNotIn("hoover", script)
        self.assertNotIn("echeck -1", script)
        self.assertLess(max(map(len, script.splitlines())), 200)
        seed_line = next(line for line in script.splitlines() if line.startswith("iseed "))
        self.assertEqual(len(seed_line.split()), 5)
        self.assertTrue(all(0 < int(value) < 2**31 for value in seed_line.split()[1:]))

    def test_npt_trial_restarts_velocities_and_uses_extended_system_pressure(self):
        script = relaxation_script("npt", 5000, 310.15, 12345, 2000, previous="nvt")
        self.assertIn("dyna leap cpt restart", script)
        self.assertIn("open read unit 11 card name nvt.rst", script)
        self.assertIn("iunread 11", script)
        self.assertIn("pcons pint pref 1.0 pmass 2000 pgamma 20", script)
        self.assertIn("hoover reft 310.15", script)
        self.assertNotIn("langevin", script)
        self.assertNotIn("berendsen", script)
        self.assertLess(max(map(len, script.splitlines())), 200)
        self.assertNotIn("iseed", script)

    def test_geometry_detects_reflection_and_core_water(self):
        model = Model(0)
        builder = MembraneBuilder(model, MembraneSpec(
            box_xy=(40.0, 40.0), box_z=80.0,
            upper_leaflet={"POPC": 1}, lower_leaflet={"POPC": 1}, clash_check=False))
        builder.prepare_topology()
        builder.add_lipid_bilayer()
        before = geometry_checks(model, builder.lipid_library, (40, 40, 80))
        self.assertEqual(before["reversed_lipids"], [])
        self.assertEqual(before["trans_lipids"], [])
        for atom in model["MEMB"].child_list[1]:
            atom.coord[0] *= -1
        waters = Solvent("WA")
        water = Residue(1, "TIP3", "SOLV")
        for name, coord in (("OH2", (0, 0, 0)), ("H1", (0.95, 0, 0)), ("H2", (-0.24, 0.93, 0))):
            water.add(Atom(name, np.array(coord, dtype=float), element="O" if name == "OH2" else "H"))
        waters.add(water)
        model.add(waters)
        after = geometry_checks(model, builder.lipid_library, (40, 40, 80))
        self.assertEqual(len(after["reversed_lipids"]), 1)
        self.assertEqual(after["waters_in_core"], 1)

    def test_reordered_crd_fails_identity_check(self):
        model = Model(0)
        waters = Solvent("WA")
        for number in (1, 2):
            residue = Residue(number, "TIP3", "SOLV")
            for name, offset in (("OH2", (0, 0, 0)), ("H1", (0.9572, 0, 0)), ("H2", (-0.24, 0.9266, 0))):
                residue.add(Atom(name, np.array(offset) + (number * 4.0, 0, 0), element="O" if name == "OH2" else "H"))
            waters.add(residue)
        model.add(waters)
        TopologyGenerator().generate_solvent(waters, "TIP3", QUIET=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_psf(model, str(directory / "system.psf"))
            write_crd(model, str(directory / "system.crd"))
            self.assertEqual(len(coordinate_pair(directory)[1]), 6)
            lines = (directory / "system.crd").read_text().splitlines()
            first = next(i for i, line in enumerate(lines) if "OH2" in line and not line.startswith("*"))
            lines[first], lines[first + 1] = lines[first + 1], lines[first]
            (directory / "system.crd").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "identities differ"):
                coordinate_pair(directory)

    def test_missing_or_changed_parameter_file_fails_before_native_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parameter = directory / "lipid.prm"
            parameter.write_text("original parameters\n")
            record = {"input_sha256": {"lipid.prm": digest(parameter)}}
            parameter.write_text("changed parameters\n")
            with self.assertRaisesRegex(ValueError, "Input hash changed"):
                charmm_worker(directory, record)
            parameter.rename(directory / "saved_lipid.prm")
            with self.assertRaises(FileNotFoundError):
                charmm_worker(directory, record)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing = directory / "result.json"
            existing.write_text("original result\n")
            with patch("sys.argv", ["runner", "--output", str(directory), "--charmm-lib", "/missing"]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(existing.read_text(), "original result\n")

    def test_changed_source_fails_before_native_load(self):
        record = {"input_sha256": {}, "source_sha256": {"scripts/membrane_pycharmm.py": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "Source hash changed"):
            charmm_worker(Path("/unused"), record)


if __name__ == "__main__":
    unittest.main()
