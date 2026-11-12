"""Compare a full transmembrane protein with the pinned PPM3 driver."""

import numpy as np
import pytest

from crimm.IO.PDBParser import PDBParser
from crimm.Modeller.MembraneOrienter import MembraneOrienter
from test_native_membrane_orientation import (
    _coords, _reference_result, native_build, reference,
)


@pytest.mark.parametrize("topology", ["in", "out"])
def test_native_matches_transmembrane_reference(reference, tmp_path, topology):
    model = PDBParser(QUIET=True).get_structure(reference / "1gzm.pdb")[0]
    for chain in list(model):
        if chain.chain_type != "Polypeptide(L)":
            model.detach_child(chain.id)
            continue
        for residue in list(chain):
            if not 1 <= residue.id[1] <= 326:
                chain.detach_child(residue.id)
            elif residue.resname == "HIS":
                residue.resname = "HSE"
    assert len(list(model.get_residues())) == 326
    assert len(list(model.get_atoms())) == 2592
    before = _coords(model)
    metrics, expected = _reference_result(reference, model, tmp_path, topology)
    result = MembraneOrienter(model, topology=topology).orient(in_place=False)
    assert result.model == 2
    np.testing.assert_array_equal(_coords(model), before)
    np.testing.assert_allclose(
        [result.score, result.membrane_thickness, result.tilt,
         result.positioning_uncertainty, result.tilt_uncertainty, result.model],
        metrics, atol=2e-5, rtol=2e-6,
    )
    np.testing.assert_allclose(
        before @ result.rotation.T + result.translation, expected, atol=2e-3, rtol=0,
    )
    np.testing.assert_allclose(result.rotation @ result.rotation.T, np.eye(3), atol=2e-6)
    assert np.isclose(np.linalg.det(result.rotation), 1, atol=2e-6)
