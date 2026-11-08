"""Keep the membrane workflow notebook clean and syntactically runnable."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from crimm.Modeller.MembraneBuilder import MembraneSpec
from crimm.Modeller.MembranePresets import get_membrane_preset, list_membrane_presets


class MembraneNotebookTests(unittest.TestCase):
    def test_code_cells_compile_without_saved_outputs(self):
        path = Path(__file__).resolve().parents[1] / "tutorials/8.PPM3MembraneWorkflow.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        for cell in code_cells:
            with self.subTest(cell=cell["id"]):
                compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_preset_table_lists_every_available_recipe(self):
        path = Path(__file__).resolve().parents[1] / "tutorials/8.PPM3MembraneWorkflow.ipynb"
        cells = {cell["id"]: "".join(cell["source"])
                 for cell in json.loads(path.read_text())["cells"]}
        self.assertIn("preset-catalogue", cells)
        tables = []
        namespace = dict(pd=pd, get_membrane_preset=get_membrane_preset,
                         list_membrane_presets=list_membrane_presets, display=tables.append)
        exec(cells["preset-catalogue"], namespace)
        table = tables[0]
        self.assertEqual(set(table.index), set(list_membrane_presets()))
        self.assertEqual(len(table), 23)
        self.assertEqual(table.loc["ERm", "PPM3 code"], "ERm")
        self.assertEqual(table.loc["DMPC", "PPM3 code"], "MPC")
        self.assertIn("endoplasmic", table.loc["ERm", "Membrane"])

    def test_selected_preset_reaches_spec_and_composition_preview(self):
        path = Path(__file__).resolve().parents[1] / "tutorials/8.PPM3MembraneWorkflow.ipynb"
        cells = {cell["id"]: "".join(cell["source"])
                 for cell in json.loads(path.read_text())["cells"]}
        self.assertIn("preset-selection", cells)
        for name, cholesterol, temperature, code, outer_chol, inner_chol in (
            ("ERm", None, 310.15, "ERm", 5.0, 5.0),
            ("PMm", None, 310.15, "PMm", 34.58, 29.0),
            ("POPC", 20, 333.15, "OPC", 20.0, 20.0),
            ("DMPC", 20, 333.15, "MPC", 20.0, 20.0),
        ):
            with self.subTest(preset=name):
                namespace = dict(np=np, pd=pd, MembraneSpec=MembraneSpec,
                                 get_membrane_preset=get_membrane_preset, display=lambda value: None)
                exec(cells["preset-selection"], namespace)
                namespace.update(membrane_name=name, cholesterol_percent=cholesterol,
                                 pc_temperature_K=333.15, initial_area_scale=1.05)
                exec(cells["settings"], namespace)
                spec = namespace["spec"]
                self.assertEqual(spec.preset, name)
                self.assertEqual(spec.initial_area_scale, 1.05)
                self.assertEqual(namespace["temperature_K"], temperature)
                self.assertEqual(namespace["ppm_membrane"], code)
                if cholesterol is not None:
                    self.assertEqual(spec.upper_lipid_ratios, {name: 80.0, "CHL1": 20.0})
                    self.assertEqual(spec.lower_lipid_ratios, {name: 80.0, "CHL1": 20.0})
                composition = namespace["requested_percent"]
                np.testing.assert_allclose(composition.sum(), 100.0)
                np.testing.assert_allclose(composition.loc["CHL1"].round(2),
                                           [outer_chol, inner_chol])


if __name__ == "__main__":
    unittest.main()
