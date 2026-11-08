"""Checks for the paper-defined OPRLM membrane catalogue."""

import pytest
import numpy as np
import io
import tarfile
from importlib.resources import files

from scripts.import_oprlm_reference import PC_TYPES, PPM_CODES, parse_recipes
from scripts.compile_oprlm_library import canonical_template, components, molecule_name, placement_metadata, reference_forcefield
from crimm.IO.PSFReader import PSFAtom, PSFData
from crimm.Modeller.MembraneBuilder import LipidConformerLibrary, MembraneBuilder, MembraneSpec
from crimm.Modeller.MembranePresets import get_membrane_preset, list_membrane_presets
from crimm.StructEntities.Model import Model
from crimm.StructEntities.Chain import Lipid
from crimm.StructEntities.Residue import Residue
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.TopoDefinitions import AtomDefinition
from crimm.Modeller.TopoLoader import TopologyGenerator, traverse_graph
from crimm.IO.PRMParser import categorize_lines, parse_line_dict
from scripts.membrane_pycharmm import geometry_checks, prepare
from types import SimpleNamespace


def source_page(rows, total):
    return "".join(
        f"<h3>{code} (reference membrane)</h3><li>Temperature (K): 310.15</li>"
        "<li>Size (A3): 80 x 80 x 100</li>"
        f"<table>{rows}<tr><td>TOTAL</td><td></td>{total}</tr></table>"
        for code in PPM_CODES
    )


def test_catalogue_keeps_all_paper_families():
    assert len(PPM_CODES) == 18
    assert PC_TYPES == ("POPC", "DOPC", "DSPC", "DMPC", "DPPC")
    assert PPM_CODES["GOLm"] == PPM_CODES["GOLf"] == "GOL"
    assert PPM_CODES["G-OM"] == "GnO"


def test_recipe_parser_keeps_unequal_leaflet_totals_and_cholesterol():
    rows = (
        "<tr><td>POPC</td><td>PC</td><td>16</td><td>7</td>"
        "<td>56.87</td><td>57.08</td><td>0</td></tr>"
        "<tr><td>CHOL</td><td>Cholesterol</td><td>37</td><td>29</td>"
        "<td>27.9</td><td>28.71</td><td>0</td></tr>"
    )
    records = parse_recipes(source_page(rows, "<td>53</td><td>36</td><td></td><td></td><td></td>"))
    assert len(records) == 18
    for record in records:
        assert record["upper_ratios"] == {"POPC": 16, "CHL1": 37}
        assert record["lower_ratios"] == {"POPC": 7, "CHL1": 29}
        assert record["reference_areas_A2"]["CHL1"] == [27.9, 28.71]
        assert record["table_totals_match"]


def test_recipe_parser_records_source_arithmetic_error_without_repair():
    rows = "<tr><td>OPPG</td><td>PG</td><td>105</td><td>60</td><td>-1</td></tr>"
    record = parse_recipes(source_page(rows, "<td>100</td><td></td><td></td>"))[0]
    assert record["upper_ratios"] == record["lower_ratios"] == {"OPPG": 105}
    assert record["table_reported_totals"] == [100, 100]
    assert record["table_row_totals"] == [105, 105]
    assert not record["table_totals_match"]


def test_recipe_parser_requires_complete_published_set():
    with pytest.raises(ValueError, match="18 published"):
        parse_recipes("<h3>ERm (ER)</h3>")


def test_reference_parameters_follow_loader_order_not_tar_member_order(tmp_path):
    path = tmp_path / "reference.tar.gz"
    files = {
        "toppar/toppar_all36_lipid_model.str": "read para card flex append\nBONDS\nC C 20 1.6\nEND\n",
        "toppar/par_all36_lipid.prm": "* base\n*\nBONDS\nC C 10 1.5\nEND\n",
        "toppar/top_all36_lipid.rtf": "* types\n*\n36 1\nMASS -1 C 12.011 C\nEND\n",
        "toppar.str": ("open read card name toppar/top_all36_lipid.rtf\n"
                       "open read card name toppar/par_all36_lipid.prm\n"
                       "stream toppar/toppar_all36_lipid_model.str\n"),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo("tarball/" + name)
            data = content.encode()
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    _, _, parameters, _ = reference_forcefield(path, {"C"})
    assert parameters.index("C C 10 1.5") < parameters.index("C C 20 1.6")


def test_compound_lipid_preserves_covalent_links_and_source_identities():
    atoms = [
        PSFAtom(1, "A", "1", "CER160", "C1", "CTL2", 0.2, 12.011),
        PSFAtom(2, "A", "1", "CER160", "O1", "OSL", -0.4, 15.9994),
        PSFAtom(3, "A", "2", "BGAL", "C1", "CC3161", 0.2, 12.011),
        PSFAtom(4, "W", "1", "TIP3", "OH2", "OT", -0.834, 15.9994),
    ]
    psf = PSFData(atoms=atoms, bonds=[(1, 2), (2, 3)], angles=[(1, 2, 3)])
    assert components(psf) == [[0, 1, 2], [3]]
    assert molecule_name(atoms[:3]) == ("GALCER", True)
    template = canonical_template(atoms[:3], [0, 1, 2], psf, compound=True)
    assert template["bonds"] == [[0, 1], [1, 2]]
    assert template["angles"] == [[0, 1, 2]]
    assert template["source_identity"] == [[0, "CER160", "C1"], [0, "CER160", "O1"], [1, "BGAL", "C1"]]
    assert len({atom[0] for atom in template["atoms"]}) == 3
    assert [atom[2] for atom in template["atoms"]] == [0.2, -0.4, 0.2]


def test_unknown_compound_lipid_is_not_split_or_guessed():
    atoms = [PSFAtom(1, "A", "1", "LIPX", "C1", "CTL2", 0, 12.011),
             PSFAtom(2, "A", "2", "GLCX", "C1", "CC3161", 0, 12.011)]
    with pytest.raises(ValueError, match="Unknown compound"):
        molecule_name(atoms)


def test_archaeal_head_selection_uses_source_atom_groups_and_charmm_digit_wildcards():
    names = ["P", "C11", "O11", "C116", "C216"]
    elements = ["P", "C", "O", "C", "C"]
    template = {"atoms": [[name, "TEST", 0, 12, element] for name, element in zip(names, elements)],
                "source_identity": [[0, "PGAR", name] for name in names],
                "bonds": [[0, 1], [0, 2], [1, 3], [3, 4]]}
    coordinates = np.array([[0, 0, z] for z in (5., 4., 6., -2., -5.)])
    restraint = ("if lipid .eq. PGAR then\ncalc zpostop = 17\n"
                 "define HEAD sele type P .or. type C1+ .or. type O1+ end\n"
                 "define TAIL sele type C216 end\nendif")
    placement_metadata("PGAR", template, coordinates, restraint)
    assert template["head_atoms"] == ["P", "C11", "O11"]
    assert template["tail_atoms"] == ["C216"]


def test_three_membered_lipid_ring_does_not_create_repeated_atom_dihedrals():
    chain = Lipid("RING")
    residue = Residue(1, "RING", "RING")
    chain.add(residue)
    atoms = [Atom(f"C{index}", np.zeros(3), element="C") for index in range(3)]
    for atom in atoms:
        residue.add(atom)
        atom.topo_definition = AtomDefinition(None, atom.name, "CTL1", 0, 12.011)
        atom.neighbors = set(other for other in atoms if other is not atom)
    angles, dihedrals = set(), set()
    traverse_graph(atoms[0], angles, dihedrals, set())
    assert len(angles) == 3
    assert not dihedrals


def test_packaged_catalogue_has_all_18_plus_five_types():
    names = list_membrane_presets()
    assert len(names) == len(set(names)) == 23
    assert set(names) == set(PPM_CODES) | set(PC_TYPES)
    for name in PPM_CODES:
        recipe = get_membrane_preset(name)
        assert recipe["ppm_membrane"] == PPM_CODES[name]
        assert len(recipe["archive_sha256"]) == 64
        assert recipe["temperature_K"] > 0
        assert recipe["whole_molecule_counts"] == [recipe["upper_ratios"], recipe["lower_ratios"]]
    assert get_membrane_preset("THYb")["upper_ratios"]["MGDGPO"] == 40
    assert get_membrane_preset("THYb")["published_upper_ratios"]["MGDG-PO"] == 45
    assert get_membrane_preset("THYb")["source_discrepancies"]


@pytest.mark.parametrize("name", PC_TYPES)
def test_pc_cholesterol_family(name):
    recipe = get_membrane_preset(name, cholesterol_percent=20)
    assert recipe["upper_ratios"] == recipe["lower_ratios"] == {name: 80, "CHL1": 20}
    assert get_membrane_preset(name, cholesterol_percent=0)["upper_ratios"] == {name: 100}
    assert recipe["temperature_K"] is None


@pytest.mark.parametrize("value", [None, -1, 100, float("nan"), float("inf"), "invalid"])
def test_invalid_cholesterol_fraction_fails(value):
    with pytest.raises(ValueError, match="cholesterol"):
        get_membrane_preset("POPC", value)


def test_presets_do_not_mutate_shared_data_or_replace_explicit_composition():
    recipe = get_membrane_preset("ERm")
    recipe["upper_ratios"].clear()
    assert get_membrane_preset("ERm")["upper_ratios"]
    with pytest.raises(ValueError, match="cholesterol override"):
        get_membrane_preset("ERm", 20)
    with pytest.raises(ValueError, match="Unknown membrane"):
        get_membrane_preset("ER")
    with pytest.raises(ValueError, match="custom compositions"):
        MembraneSpec.from_preset("ERm", box_xy=(80, 80), box_z=100, lipid_ratios={"POPC": 1})


def test_membrane_reference_does_not_export_protein_cmap_or_backbone_parameters():
    builder = MembraneBuilder(Model(0), MembraneSpec.from_preset("ERm", box_xy=(80, 80), box_z=100))
    parameters = builder.topology.param_dict["lipid"].param_dict
    assert not parameters["cmap"]
    assert ("NH1", "CT1") not in parameters["bonds"]


def test_shared_nonbonded_parameters_load_once_and_conflicts_fail_before_export(tmp_path):
    topology = TopologyGenerator()
    topology.load_residue_definitions("Polypeptide(L)")
    library = LipidConformerLibrary(files("crimm.Data").joinpath("oprlm_lipids.npz"))
    library.install_topology(topology, "ERm")
    topology.write_toppar(path=str(tmp_path / "compatible"))
    membrane = parse_line_dict(categorize_lines((tmp_path / "compatible/toppar/lipid.prm").read_text().splitlines()))
    assert "H" not in membrane["nonbonded"]
    protein = topology.param_dict["protein"].param_dict
    assert protein["nonbonded"]["H"] == topology.param_dict["lipid"].param_dict["nonbonded"]["H"]
    protein["nonbonded"]["H"] = protein["nonbonded"]["H"]._replace(epsilon=-0.1)
    with pytest.raises(ValueError, match="Incompatible shared"):
        topology.write_toppar(path=str(tmp_path / "incompatible"))
    assert not (tmp_path / "incompatible").exists()


def test_preset_retains_leaflet_specific_areas():
    spec = MembraneSpec.from_preset("PMm", box_xy=(80, 80), box_z=100)
    assert spec.upper_lipid_ratios["CHL1"] == 37
    assert spec.lower_lipid_ratios["CHL1"] == 29
    preset = get_membrane_preset("PMm")
    for side, original_cholesterol_area in (("upper", 27.9), ("lower", 28.71)):
        areas = getattr(spec, f"{side}_lipid_area_overrides")
        counts = preset[f"{side}_ratios"]
        assert sum(count * areas[name] for name, count in counts.items()) == pytest.approx(75.26 ** 2)
        assert areas["CHL1"] / areas["POPC"] == pytest.approx(
            original_cholesterol_area / preset["reference_areas_A2"]["POPC"][0 if side == "upper" else 1])


def test_archaeal_source_head_offsets_follow_requested_thickness():
    with pytest.warns(UserWarning, match="experimental"):
        spec = MembraneSpec.from_preset("aPM", box_xy=(81.04, 81.04), box_z=100)
    builder = MembraneBuilder(Model(0), spec)
    assert builder._charmm_gui_head_z("PGAR") == 18
    assert builder._charmm_gui_head_z("AROL") == 16
    assert builder._charmm_gui_head_z("MEN8") == 18
    builder.spec.lipid_z += 5
    assert builder._charmm_gui_head_z("PGAR") == 23
    assert builder._charmm_gui_head_z("AROL") == 21


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), None])
def test_pc_example_requires_a_valid_explicit_temperature(tmp_path, temperature):
    args = SimpleNamespace(seed=12345, membrane="POPC", cholesterol_percent=20,
                           upper_lipid_ratios=None, lower_lipid_ratios=None,
                           ppm_membrane=None, temperature=temperature)
    with pytest.raises(ValueError, match="temperature|Temperature"):
        prepare(tmp_path, args, {})


def test_cholesterol_option_without_a_preset_is_not_silently_ignored(tmp_path):
    args = SimpleNamespace(seed=12345, membrane=None, cholesterol_percent=20)
    with pytest.raises(ValueError, match="requires.*preset"):
        prepare(tmp_path, args, {})


def test_selected_conformers_preserve_stereochemistry_and_double_bond_states():
    library = LipidConformerLibrary(files("crimm.Data").joinpath("oprlm_lipids.npz"))
    for name, template in library.templates.items():
        all_coords = library._coords_by_resname[name]
        assert len(all_coords) == template["source_conformer_count"] + ("generated_conformer" in template)
        coords = all_coords[template["valid_conformers"]]
        assert len(coords)
        indices = {name: index for index, name in enumerate(library._atoms_by_resname[name])}
        for centre, neighbours in template["chiral_centres"].items():
            vectors = coords[:, [indices[atom] for atom in neighbours]] - coords[:, indices[centre], None]
            determinants = np.linalg.det(vectors)
            assert np.all(np.sign(determinants) == np.sign(determinants[0])), (name, centre)
        for guard in template["dihedral_guards"]:
            if guard["kind"] != "double_bond":
                continue
            a, b, c, d = np.moveaxis(coords[:, [indices[atom] for atom in guard["names"]]], 1, 0)
            axis = c - b
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            first, last = a - b, d - c
            first -= np.sum(first * axis, axis=1, keepdims=True) * axis
            last -= np.sum(last * axis, axis=1, keepdims=True) * axis
            cosine_sign = np.sign(np.sum(first * last, axis=1))
            assert np.all(cosine_sign == (1 if guard["angle"] == 0 else -1)), (name, guard)
    assert {name for name, template in library.templates.items() if "generated_conformer" in template} == {
        "LLPA", "LLPE", "LLPS", "PLPI"}


def assert_reference_packing(name, initial_area_scale=1.0, seed=12345):
    preset = get_membrane_preset(name)
    cell = preset["reference_box_A"]
    builder = MembraneBuilder(Model(0), MembraneSpec.from_preset(
        name, box_xy=cell[:2], box_z=cell[2], max_repack_attempts=200,
        initial_area_scale=initial_area_scale, random_seed=seed))
    assert builder.upper_leaflet == preset["upper_ratios"]
    assert builder.lower_leaflet == preset["lower_ratios"]
    builder.add_lipid_bilayer()
    np.testing.assert_allclose(builder._box_dims[:2], np.array(cell[:2]) * np.sqrt(initial_area_scale))
    geometry = geometry_checks(builder.model, builder.lipid_library, builder._box_dims)
    assert geometry["actual_leaflet_composition"] == [builder.upper_leaflet, builder.lower_leaflet]
    for key in ("reversed_lipids", "trans_lipids", "inward_lipid_heads", "ring_penetrations",
                "periodic_heavy_contacts_below_1_2_A"):
        assert not geometry[key], (name, key, geometry[key])


@pytest.mark.slow
@pytest.mark.parametrize("name", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=ValueError,
        reason="Experimental aPM reference density: final MEN8 placement exhausts 200 conformer retries."))
    if name == "aPM" else name for name in PPM_CODES
])
def test_reference_box_composition_packs_without_geometry_errors(name):
    assert_reference_packing(name)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [12345, 22345, 32345])
def test_expanded_archaeal_recipe_preserves_counts_and_geometry(seed):
    assert_reference_packing("aPM", initial_area_scale=1.05, seed=seed)


@pytest.fixture(params=tuple(PPM_CODES))
def preset_lipids(request):
    path = files("crimm.Data").joinpath("oprlm_lipids.npz")
    recipe = get_membrane_preset(request.param)
    names = recipe["upper_ratios"].keys() | recipe["lower_ratios"].keys()
    builder = MembraneBuilder(Model(0), MembraneSpec(
        box_xy=(300, 300), box_z=160, upper_leaflet={name: 1 for name in names},
        lower_leaflet={}, lipid_library_path=str(path), preset=request.param, clash_check=False))
    builder.add_lipid_bilayer()
    return builder


def test_all_reference_molecules_have_complete_coordinates_and_parameters(preset_lipids, tmp_path):
    builder = preset_lipids
    assert len(builder.lipid_library.templates) == 105
    residues = list(builder.model.get_residues())
    assert len(residues) == len(builder.upper_leaflet)
    assert all(residue.resname != "TIP3" for residue in residues)
    for chain in builder.model:
        assert not chain.topology.missing_param_dict
    geometry = geometry_checks(builder.model, builder.lipid_library, (300, 300, 160))
    assert not geometry["reversed_lipids"] and not geometry["trans_lipids"]
    assert not geometry["inward_lipid_heads"] and not geometry["ring_penetrations"]
    for residue in residues:
        template = builder.lipid_library.templates[residue.resname]
        assert set(residue.child_dict) == {atom[0] for atom in template["atoms"]}
        assert not residue.missing_atoms and not residue.missing_hydrogens
        np.testing.assert_allclose([atom.topo_definition.charge for atom in residue],
                                   [atom[2] for atom in template["atoms"]], rtol=0, atol=1e-12)
        chain_type = "Sterol" if residue.parent.chain_type == "Sterol" else "Lipid"
        heads = MembraneBuilder._head_atoms(residue, chain_type)
        tails = MembraneBuilder._tail_atoms(residue, chain_type)
        assert heads and tails
        np.testing.assert_allclose(np.mean([atom.coord[2] for atom in heads]), 20, atol=1e-8, rtol=0)
        assert np.mean([a.coord[2] for a in heads]) > np.mean([a.coord[2] for a in tails])
        indices = {id(residue[atom[0]]): index for index, atom in enumerate(template["atoms"])}
        for key in ("bonds", "angles", "dihedrals", "impropers", "cmap"):
            def canonical(term):
                term = tuple(term)
                return min(term, term[::-1])
            generated = {canonical(indices[id(atom)] for atom in term)
                         for term in getattr(residue.parent.topology, key, ())
                         if id(term[0]) in indices}
            expected = {canonical(term) for term in template[key]}
            assert generated == expected, (residue.resname, key, generated - expected, expected - generated)
    builder.topology.write_toppar(path=str(tmp_path))
    mass_types = {line.split()[2] for line in (tmp_path / "toppar/membrane_types.prm").read_text().splitlines()
                  if line.startswith("MASS")}
    assert {atom.topo_definition.atom_type for atom in builder.model.get_atoms()} <= mass_types
