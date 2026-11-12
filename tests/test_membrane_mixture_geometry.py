"""Deliberate stereochemistry and ring-penetration corruption checks."""

import argparse

import networkx as nx
import numpy as np
import pytest

from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.StructEntities.Model import Model
from crimm.charmm_gui_9qsy.pentest import check_ring_penetration
from scripts.membrane_pycharmm import CHIRAL_CENTRES, STEROL_CHIRAL_SIGNS, dihedral_angle, geometry_checks, lipid_ratios


@pytest.mark.parametrize("resname", MembraneBuilder.LIPID_TAILS)
def test_each_lipid_preserves_chirality_and_head_direction(resname):
    model = Model(0)
    builder = MembraneBuilder(model, MembraneSpec(box_xy=(40,40), box_z=80,
        upper_leaflet={resname: 1}, lower_leaflet={}, clash_check=False, random_seed=12345))
    builder.add_lipid_bilayer()
    before = geometry_checks(model, builder.lipid_library, (40,40,80))
    assert not before["reversed_lipids"] and not before["trans_lipids"]
    assert not before["inward_lipid_heads"] and not before["ring_penetrations"]
    assert before["actual_leaflet_composition"] == [{resname: 1}, {}]
    if resname == "CHL1":
        coords = {atom.name: atom.coord for atom in model.get_atoms()}
        for centre, (first, second, last) in CHIRAL_CENTRES[resname].items():
            angle = dihedral_angle([coords[name] for name in (first, second, centre, last)])
            assert np.sign(angle) == STEROL_CHIRAL_SIGNS[centre]
            assert abs(abs(angle) - 120) < 25
    for atom in model.get_atoms():
        atom.coord[0] *= -1
    after = geometry_checks(model, builder.lipid_library, (40,40,80))
    assert len(after["reversed_lipids"]) == (8 if resname == "CHL1" else 1)


@pytest.mark.parametrize("periodic", [False, True])
def test_ring_check_detects_penetration_and_first_periodic_image(periodic):
    graph = nx.Graph()
    coords = {0: np.array([-9.9 if periodic else 0,0,-1.0]),
              1: np.array([-9.9 if periodic else 0,0,1.0]),
              2: np.array([-1,-1,0.0]), 3: np.array([1,-1,0.0]),
              4: np.array([1,1,0.0]), 5: np.array([-1,1,0.0])}
    for node in coords:
        graph.add_node(node, segid="TEST", resid="1" if node < 2 else "2", resname="TEST", name=str(node))
    graph.add_edges_from([(0,1),(2,3),(3,4),(4,5),(5,2)])
    pairs, rings = check_ring_penetration(graph, coords, pbc=[10,10] if periodic else [])
    assert len(pairs) == 1 and set(pairs[0]) == {0,1}
    assert set(rings[0]) == {2,3,4,5}
    coords[0][1] = coords[1][1] = 4
    assert check_ring_penetration(graph, coords, pbc=[10,10] if periodic else []) == ([], [])


@pytest.mark.parametrize("value", ['{"POPC":1,"POPC":2}', '[1,2]', '{bad'])
def test_ratio_cli_rejects_ambiguous_input(value):
    with pytest.raises(argparse.ArgumentTypeError):
        lipid_ratios(value)
