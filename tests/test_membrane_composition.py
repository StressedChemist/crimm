"""Independent molecular proportions for the two leaflets."""

import numpy as np
import pytest
from dataclasses import replace

from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.StructEntities.Model import Model


def builder(**kwargs):
    return MembraneBuilder(Model(0), MembraneSpec(
        box_xy=(40, 40), box_z=80, use_lipid_library=False, **kwargs))


def test_independent_leaflet_ratios_and_metadata():
    membrane = builder(upper_lipid_ratios={"POPC": 80, "CHL1": 20},
                       lower_lipid_ratios={"DMPC": 1})
    assert membrane.upper_leaflet == {"POPC": 22, "CHL1": 5}
    assert membrane.lower_leaflet == {"DMPC": 27}
    assert membrane.leaflet_composition["upper"]["requested_percent"] == {"POPC": 80, "CHL1": 20}
    assert sum(membrane.leaflet_composition["upper"]["realized_percent"].values()) == pytest.approx(100)


@pytest.mark.parametrize("scale", [1.0, 1.05, 1.10])
def test_initial_area_scale_keeps_nominal_counts_and_does_not_mutate_spec(scale):
    spec = MembraneSpec(box_xy=(40, 50), box_z=80, solvation_box=(40, 50, 90),
                        upper_lipid_ratios={"POPC": 8, "CHL1": 2},
                        lower_lipid_ratios={"DMPC": 1}, use_lipid_library=False)
    nominal = MembraneBuilder(Model(0), spec)
    expanded = MembraneBuilder(Model(0), replace(spec, initial_area_scale=scale))
    assert expanded.upper_leaflet == nominal.upper_leaflet
    assert expanded.lower_leaflet == nominal.lower_leaflet
    assert expanded.leaflet_composition == nominal.leaflet_composition
    np.testing.assert_allclose(expanded.spec.box_xy, np.array(spec.box_xy) * np.sqrt(scale))
    np.testing.assert_allclose(expanded._box_dims, (*expanded.spec.box_xy, 90))
    assert expanded.spec.box_z == 80
    assert spec.box_xy == (40, 50) and spec.solvation_box == (40, 50, 90)


@pytest.mark.parametrize("scale", [0, -1, 0.99, np.nan, np.inf])
def test_invalid_initial_area_scale_fails_before_model_mutation(scale):
    model = Model(0)
    with pytest.raises(ValueError, match="initial_area_scale"):
        MembraneBuilder(model, MembraneSpec(box_xy=(40, 40), box_z=80,
            upper_leaflet={}, lower_leaflet={}, initial_area_scale=scale))
    assert not hasattr(model, "topology_loader")


def test_initial_area_scale_preserves_intramolecular_geometry_and_head_depth():
    spec = MembraneSpec(box_xy=(40, 40), box_z=80, upper_leaflet={"POPC": 1},
                        lower_leaflet={}, clash_check=False)
    coordinates = []
    for scale in (1.0, 1.10):
        membrane = MembraneBuilder(Model(0), replace(spec, initial_area_scale=scale))
        membrane.add_lipid_bilayer()
        residue = next(membrane.model.get_residues())
        coordinates.append(np.array([atom.coord for atom in residue]))
        assert residue["P"].coord[2] == pytest.approx(spec.lipid_z)
    first, second = coordinates
    np.testing.assert_allclose(first - first[0], second - second[0], atol=1e-10, rtol=0)


def test_initial_area_scale_expands_packed_heads_without_changing_z(tmp_path):
    path = tmp_path / "heads.crd"
    path.write_text("* heads\n*\n2\n1 1 POPC P 1 2 3 MEMB 1 0\n2 2 POPC P -1 -2 -3 MEMB 2 0\n")
    membrane = builder(upper_leaflet={"POPC": 1}, lower_leaflet={"POPC": 1},
        initial_area_scale=1.10, use_charmm_gui_head_positions=True,
        charmm_gui_head_crd=str(path))
    positions = np.array([head.coord for head in membrane.packed_head_positions])
    np.testing.assert_allclose(positions[:, :2], np.array([[1, 2], [-1, -2]]) * np.sqrt(1.10))
    np.testing.assert_allclose(positions[:, 2], [3, -3])


def test_relative_scales_and_legacy_tie_order():
    for ratios in ({"POPC": 8, "CHL1": 2}, {"POPC": .8, "CHL1": .2}):
        assert builder(lipid_ratios=ratios).upper_leaflet == {"POPC": 22, "CHL1": 5}
    assert MembraneBuilder._counts_from_ratios(1, {"DMPC": .5, "POPC": .5}) == {"DMPC": 1, "POPC": 0}


def test_counts_override_global_ratios_and_empty_leaflet_stays_empty():
    membrane = builder(upper_leaflet={}, lipid_ratios={"POPC": 1},
                       lower_lipid_ratios={"DOPC": 1})
    assert membrane.upper_leaflet == {}
    assert membrane.lower_leaflet == {"DOPC": 24}
    assert membrane.leaflet_composition["upper"]["realized_percent"] == {}


@pytest.mark.parametrize("ratios", [
    {}, {"POPC": 0}, {"POPC": -1, "DOPC": 2}, {"POPC": np.nan},
    {"POPC": np.inf}, {"POPC": "bad"}, {"popc": 1, "POPC": 2},
])
def test_invalid_ratios_fail_before_placement(ratios):
    with pytest.raises(ValueError):
        builder(lipid_ratios=ratios)


@pytest.mark.parametrize("counts", [{"POPC": 1.9}, {"POPC": np.nan},
                                    {"POPC": -1}, {"popc": 1, "POPC": 1}])
def test_invalid_counts_do_not_get_truncated(counts):
    with pytest.raises(ValueError):
        builder(upper_leaflet=counts, lower_leaflet={})


def test_conflicting_leaflet_inputs_fail():
    with pytest.raises(ValueError, match="both"):
        builder(upper_leaflet={"POPC": 1}, upper_lipid_ratios={"POPC": 1}, lower_leaflet={})


def test_zero_weight_unknown_lipid_is_ignored():
    assert builder(lipid_ratios={"POPC": 1, "UNKNOWN": 0}).upper_leaflet == {"POPC": 25}


@pytest.mark.parametrize("area", [0, -1, np.nan, np.inf])
def test_invalid_area_is_rejected(area):
    with pytest.raises(ValueError):
        builder(lipid_ratios={"POPC": 1}, area_per_lipid=area)


def test_missing_conformer_fails_before_model_mutation():
    model = Model(0)
    membrane = MembraneBuilder(model, MembraneSpec(box_xy=(40, 40), box_z=80,
        upper_leaflet={"POPC": 1, "POPE": 1}, lower_leaflet={}))
    with pytest.raises(ValueError, match="POPE"):
        membrane.add_lipid_bilayer()
    assert not list(model.get_residues())


@pytest.mark.parametrize("lipid,tails", [
    ("DMPC", {"C214", "C314"}), ("DPPC", {"C216", "C316"}),
    ("DSPC", {"C218", "C318"}), ("POPC", {"C218", "C316"}),
    ("DOPC", {"C218", "C318"}), ("CHL1", {"C26", "C27"}),
])
def test_each_packaged_lipid_uses_its_tail_ends(lipid, tails):
    membrane = MembraneBuilder(Model(0), MembraneSpec(box_xy=(40, 40), box_z=80,
        upper_leaflet={lipid: 1}, lower_leaflet={}, clash_check=False))
    membrane.prepare_topology()
    membrane.add_lipid_bilayer()
    residue = next(membrane.model.get_residues())
    assert {a.name for a in membrane._tail_atoms(residue, "Sterol" if lipid == "CHL1" else "Lipid")} == tails
