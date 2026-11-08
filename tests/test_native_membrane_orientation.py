"""Native PPM3 checks against a pinned, separately compiled reference."""

import hashlib
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from crimm.IO.PDBParser import PDBParser
from crimm.Modeller.MembraneOrienter import MembraneOrienter
from crimm.Modeller.TopoFixer import build_hydrogens_for_chain, build_missing_atoms_for_chain
from crimm.Modeller.TopoLoader import TopologyGenerator
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Chain import Heterogens
from crimm.StructEntities.Residue import Residue


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ARCHIVE = Path(__file__).parent / "data" / "ppm3_reference.tar.gz"
PDB_ARCHIVE = Path(__file__).parent / "data" / "ppm2_reference.tar.gz"
REFERENCE_SHA256 = "ca599b95e7ec977abe9bee778cd6444daf2d89dbffd6ed537e4da9d0d3f22f46"
SOURCES = (
    "opm.f", "min.f", "rmsd.f", "solva.f", "readpdb.f", "read_small.f",
    "watface.f", "tilting.f", "locate.f", "profile.f", "hbcor.f",
    "deftm.f", "orient.f", "find_segm.f",
)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module", autouse=True)
def native_build():
    assert shutil.which("gfortran"), "The native PPM3 tests require gfortran."
    _run([sys.executable, "setup.py", "build_ext", "--inplace", "--force"], cwd=ROOT)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    assert hashlib.sha256(REFERENCE_ARCHIVE.read_bytes()).hexdigest() == REFERENCE_SHA256
    source = tmp_path_factory.mktemp("ppm3-reference")
    with tarfile.open(REFERENCE_ARCHIVE) as archive:
        for name in (*SOURCES, "res.lib"):
            (source / name).write_bytes(archive.extractfile(name).read())
    with tarfile.open(PDB_ARCHIVE) as archive:
        for name in ("1rsy.pdb", "1gzm.pdb"):
            (source / name).write_bytes(archive.extractfile(name).read())
    path = source / "find_segm.f"
    text = path.read_text()
    text = text.replace("if(i.gt.nres-2)", "if(i.gt.nca-2)")
    text = text.replace("      do i=1,nsegm-1", "      ichange=0\n      do i=1,nsegm-1")
    path.write_text(text)
    for name in ("watface.f", "hbcor.f"):
        path = source / name
        text = path.read_text().replace("ang=acos(co)", "ang=acos(max(-1.,min(1.,co)))")
        for expression in ("(r12*r12+r23*r23-r13)/(2.*r12*r23)",
                           "(r23*r23+r34*r34-r24)/(2.*r23*r34)"):
            text = text.replace(f"acos({expression})", f"acos(max(-1.,min(1.,{expression})))")
        path.write_text(text)
    path = source / "tilting.f"
    text = path.read_text()
    start = text.index("      subroutine project ")
    text = text[:start] + text[start:].replace(
        "      smax=0.", "      phimax=0.\n      yzmax=0.\n      smax=0.", 1,
    )
    path.write_text(text)
    path = source / "opm.f"
    text = path.read_text()
    start = "      open (21,file=pdbtempl)"
    assert text.count(start) == 2
    text = text.replace(start, "      nlipid=0\n" + start, 1)
    assert text.count("      nsuper=10") == 1
    text = text.replace("      nsuper=10", "      write(*,'(a,i10)') 'PPM3_ATOMS ',nat\n      nsuper=10")
    marker = "c     Define list of names of subunits:"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        "      write(*,'(a,6es24.15)') 'PPM3_REFERENCE ',\n"
        "     *  ener,dmin,tilt,d12,t12,real(model)\n" + marker,
    )
    path.write_text(text)
    _run([
        "gfortran", "-O2", "-std=legacy", "-ffixed-line-length-none",
        "-o", "immers", *SOURCES,
    ], cwd=source)
    return source


def _model(reference):
    model = PDBParser(QUIET=True).get_structure(reference / "1rsy.pdb")[0]
    for chain in list(model):
        if chain.chain_type != "Polypeptide(L)":
            model.detach_child(chain.id)
            continue
        for residue in list(chain):
            if not 193 <= residue.id[1] <= 264:
                chain.detach_child(residue.id)
            elif residue.resname == "HIS":
                residue.resname = "HSE"
    assert len(list(model.get_residues())) == 72
    assert sum(r.resname == "HSE" for r in model.get_residues()) == 3
    return model


def _coords(model):
    return np.asarray([atom.coord for atom in model.get_atoms()], dtype=float)


def _reference_result(reference, model, workdir, topology, membrane_type="OPC"):
    shutil.copyfile(reference / "res.lib", workdir / "res.lib")
    records = []
    serial = 0
    atom_order = []
    for chain_index, chain in enumerate(model):
        for residue in chain:
            residue_index = residue.id[1]
            assert residue.id[2] == " " and -999 <= residue_index <= 9999
            native_name = "HIS" if residue.resname == "HSE" else residue.resname
            for atom in sorted(residue, key=lambda a: (a.name != "N", a.name)):
                serial += 1
                atom_order.append(atom)
                x, y, z = atom.coord
                record = (
                    f"ATOM  {serial:5d}  {atom.name:<3s} {native_name:3s} "
                    f"{chr(65 + chain_index)}{residue_index:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                )
                assert record[13:17] == atom.name.ljust(4)
                records.append(record)
    (workdir / "protein.pdb").write_text("".join(records) + "END\n")
    result = _run(
        [str(reference / "immers")], cwd=workdir,
        input=f"2\nno\nprotein.pdb\n1\n{membrane_type}\nplanar\n{topology}\nA\n", timeout=300,
    )
    atom_count = next(line for line in result.stdout.splitlines() if line.startswith("PPM3_ATOMS "))
    assert int(atom_count.split()[1]) == serial
    metrics = next(line for line in result.stdout.splitlines() if line.startswith("PPM3_REFERENCE "))
    metrics = np.asarray([float(value) for value in metrics.split()[1:]])
    output = (workdir / "proteinout.pdb").read_text().splitlines()
    coordinates = np.asarray([
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in output
        if line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] != "DUM")
    ])
    by_atom = {id(atom): coord for atom, coord in zip(atom_order, coordinates)}
    return metrics, np.asarray([by_atom[id(atom)] for atom in model.get_atoms()])


@pytest.mark.parametrize("topology,membrane_type", [("in", "OPC"), ("out", "OPC"), ("in", "MPC"), ("out", "PMm")])
def test_native_matches_independent_reference(reference, tmp_path, topology, membrane_type):
    model = _model(reference)
    before = _coords(model)
    metrics, expected = _reference_result(reference, model, tmp_path, topology, membrane_type)
    result = MembraneOrienter(model, topology=topology, membrane_type=membrane_type).orient(in_place=False)
    np.testing.assert_array_equal(_coords(model), before)
    np.testing.assert_allclose(
        [result.score, result.membrane_thickness, result.tilt,
         result.positioning_uncertainty, result.tilt_uncertainty, result.model],
        metrics, atol=2e-5, rtol=2e-6,
    )
    np.testing.assert_allclose(before @ result.rotation.T + result.translation, expected, atol=2e-3, rtol=0)
    np.testing.assert_allclose(result.rotation @ result.rotation.T, np.eye(3), atol=2e-6)
    assert np.linalg.det(result.rotation) == pytest.approx(1, abs=2e-6)
    assert result.ph == 7.5 and result.temperature == 293.0
    assert result.profile == "DOPC"
    assert result.native_version == "PPM3" and result.membrane_type == membrane_type
    assert result.matching_thickness > 0 and result.mismatch_coefficient > 0
    assert result.topology_reference["atom"] == "N"
    assert result.topology_reference["final_z"] * (1 if topology == "out" else -1) > 0
    assert result.scored_atoms == "protein core heavy atoms"


def test_native_transforms_full_assembly_without_replacing_state(reference):
    model = _model(reference)
    chain = model.child_list[0]
    first, last = chain.child_list[0], chain.child_list[-1]
    TopologyGenerator().generate(chain, first_patch="ACE", last_patch="CT3", QUIET=True)
    build_missing_atoms_for_chain(chain)
    build_hydrogens_for_chain(chain)
    assert first.topo_definition.patch_with == "ACE" and last.topo_definition.patch_with == "CT3"
    assert {"CAY", "CY", "OY"} <= set(first.child_dict)
    assert {"NT", "CAT"} <= set(last.child_dict)
    assert all("HE2" in r and "HD1" not in r for r in chain if r.resname == "HSE")
    ligand = Heterogens("LIGAND")
    residue = Residue(("H_LIG", 7, "B"), "LIG", "LIGA")
    residue.add(Atom("C1", np.array([1.0, 2.0, 3.0]), element="C"))
    ligand.add(residue)
    model.add(ligand)
    original_atoms = list(model.get_atoms())
    original_labels = [(r.id, r.resname, r.segid) for r in model.get_residues()]
    original_definitions = [r.topo_definition for r in model.get_residues()]
    atom_definitions = [atom.topo_definition for atom in original_atoms]
    charges = [definition.charge if definition else None for definition in atom_definitions]
    topology = chain.topology
    before = _coords(model)
    result = MembraneOrienter(model).orient()
    assert all(a is b for a, b in zip(model.get_atoms(), original_atoms))
    assert [(r.id, r.resname, r.segid) for r in model.get_residues()] == original_labels
    assert all(r.topo_definition is d for r, d in zip(model.get_residues(), original_definitions))
    assert chain.topology is topology
    assert all(atom.topo_definition is d for atom, d in zip(model.get_atoms(), atom_definitions))
    assert [atom.topo_definition.charge if atom.topo_definition else None for atom in model.get_atoms()] == charges
    np.testing.assert_allclose(_coords(model), before @ result.rotation.T + result.translation, atol=2e-6)
    np.testing.assert_allclose(
        np.linalg.norm(_coords(model) - _coords(model)[0], axis=1),
        np.linalg.norm(before - before[0], axis=1), atol=5e-5,
    )


def test_native_a_b_a_has_no_state_leak(reference):
    model = _model(reference)
    first = MembraneOrienter(model).orient(in_place=False)
    second_model = _model(reference)
    for atom in second_model.get_atoms():
        atom.coord = atom.coord + np.array([250, -100, 50])
    MembraneOrienter(second_model, ph=4.5, temperature=305.0, membrane_type="PMm").orient(in_place=False)
    last = MembraneOrienter(model).orient(in_place=False)
    np.testing.assert_array_equal(first.rotation, last.rotation)
    np.testing.assert_array_equal(first.translation, last.translation)
    assert first.score == last.score
    assert first.membrane_thickness == last.membrane_thickness


def test_native_ambiguous_topology_reference_does_not_mutate(reference, monkeypatch):
    import ctypes
    module = importlib.import_module("crimm.Modeller.MembraneOrienter")
    model = _model(reference)
    before = _coords(model)
    def native(n, coords, names, residues, chains, ids, path, path_length, ph, temperature,
               topology, system, rotation, translation, metrics, matching, status):
        rotation[:] = np.eye(3)
        translation[:] = [0, 0, -coords[0, 2]]
        metrics[:] = [-10, 28.8, 0, 1, 1, 1]
        matching[:] = [28.8, 28.8, 0.02]
        ctypes.cast(status, ctypes.POINTER(ctypes.c_int)).contents.value = 0
    monkeypatch.setattr(module, "_native_function", lambda: native)
    with pytest.raises(RuntimeError, match="midplane"):
        MembraneOrienter(model).orient()
    np.testing.assert_array_equal(_coords(model), before)


@pytest.mark.parametrize("state", ["HIS", "HSD", "HSP"])
def test_native_rejects_unresolved_or_unsupported_histidine(reference, state):
    model = _model(reference)
    next(r for r in model.get_residues() if r.resname == "HSE").resname = state
    before = _coords(model)
    with pytest.raises(ValueError, match=state):
        MembraneOrienter(model).orient()
    np.testing.assert_array_equal(_coords(model), before)


@pytest.mark.parametrize("missing", ["CG", "ND1", "CD2", "CE1", "NE2"])
def test_native_requires_complete_histidine_ring(reference, missing):
    model = _model(reference)
    next(r for r in model.get_residues() if r.resname == "HSE").detach_child(missing)
    with pytest.raises(ValueError, match=missing):
        MembraneOrienter(model).orient()


def test_native_rejects_contradictory_histidine_hydrogen(reference):
    model = _model(reference)
    residue = next(r for r in model.get_residues() if r.resname == "HSE")
    residue.add(Atom("HD1", residue["ND1"].coord + [1, 0, 0], element="H"))
    with pytest.raises(ValueError, match="HD1"):
        MembraneOrienter(model).orient()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_native_rejects_nonfinite_assembly_coordinates(reference, value):
    model = _model(reference)
    next(model.get_atoms()).coord[0] = value
    with pytest.raises(ValueError, match="finite"):
        MembraneOrienter(model).orient()


def test_native_rejects_nonfinite_retained_cofactor_without_mutation(reference):
    model = _model(reference)
    before = _coords(model)
    ligand = Heterogens("LIGAND")
    residue = Residue(("H_LIG", 7, " "), "LIG", "LIGA")
    residue.add(Atom("C1", np.array([np.nan, 2.0, 3.0]), element="C"))
    ligand.add(residue)
    model.add(ligand)
    with pytest.raises(ValueError, match="finite"):
        MembraneOrienter(model).orient(in_place=False)
    np.testing.assert_array_equal(_coords(model.child_list[0]), before)


def test_native_rejects_nonstandard_side_chain_patch(reference):
    model = _model(reference)
    residue = next(r for r in model.get_residues() if r.resname == "ASP")
    residue.topo_definition = SimpleNamespace(patch_with="ASPP")
    with pytest.raises(ValueError, match="ASPP"):
        MembraneOrienter(model).orient()


def test_native_rejects_untyped_heavy_atom(reference):
    model = _model(reference)
    residue = next(model.get_residues())
    residue.add(Atom("QQ", residue["CA"].coord + [2, 0, 0], element="C"))
    with pytest.raises(ValueError, match="QQ"):
        MembraneOrienter(model).orient()


def test_native_preserves_insertion_codes_and_sequence_gaps(reference):
    model = _model(reference)
    chain = model.child_list[0]
    chain.id = "LONG_AUTHOR_CHAIN"
    for residue in list(chain)[3:]:
        chain.detach_child(residue.id)
    for residue, identifier in zip(chain, ((" ", 50000, "A"), (" ", 50000, "B"), (" ", 50005, " "))):
        residue.id = identifier
    labels = [(chain.id, r.id, r.resname) for r in chain]
    _, _, _, native_chains, native_ids = MembraneOrienter(model)._protein_input()
    borders = np.cumsum([0, *(len(r) for r in chain)])[:-1]
    np.testing.assert_array_equal(native_ids[borders], [50000, 50001, 50006])
    assert len(set(native_chains)) == 1
    assert [(chain.id, r.id, r.resname) for r in chain] == labels


def test_native_preserves_matching_residue_numbers_across_chains(reference):
    model = _model(reference)
    chain = model.child_list[0]
    for residue in list(chain)[3:]:
        chain.detach_child(residue.id)
    second = chain.copy()
    second.id = "SECOND_LONG_CHAIN"
    model.add(second)
    _, _, _, native_chains, native_ids = MembraneOrienter(model)._protein_input()
    first_borders = np.cumsum([0, *(len(r) for r in chain)])[:-1]
    second_borders = first_borders + sum(len(r) for r in chain)
    np.testing.assert_array_equal(native_ids[first_borders], [193, 194, 195])
    np.testing.assert_array_equal(native_ids[second_borders], [193, 194, 195])
    assert native_chains[first_borders[0]] != native_chains[second_borders[0]]


def test_native_rejects_internal_terminal_atoms(reference):
    model = _model(reference)
    residue = model.child_list[0].child_list[2]
    residue.add(Atom("CAY", residue["CA"].coord + [1, 0, 0], element="C"))
    with pytest.raises(ValueError, match="CAY"):
        MembraneOrienter(model).orient()


@pytest.mark.parametrize("parameter,value", [
    ("ph", np.nan), ("ph", -1), ("ph", 15),
    ("temperature", np.inf), ("temperature", 0), ("topology", "unknown"),
    ("membrane_type", "MIC"), ("membrane_type", ""), ("membrane_type", "opc"),
])
def test_native_rejects_invalid_model_conditions(reference, parameter, value):
    with pytest.raises(ValueError):
        MembraneOrienter(_model(reference), **{parameter: value})


@pytest.mark.parametrize("kind,limit", [("chains", 60), ("residues", 20000), ("atoms", 150000)])
def test_native_rejects_input_capacity_before_native_call(reference, kind, limit):
    model = _model(reference)
    chain = model.child_list[0]
    if kind == "chains":
        excessive = [chain] * (limit + 1)
    else:
        class RepeatedChain(list):
            chain_type = "Polypeptide(L)"
        residue = next(r for r in chain if r.resname == ("GLY" if kind == "residues" else "TRP"))
        count = limit + 1 if kind == "residues" else limit // len(residue) + 1
        excessive = [RepeatedChain([residue] * count)]
    with pytest.raises(ValueError, match=str(limit)):
        MembraneOrienter(excessive)._protein_input()


def test_native_unavailable_library_has_no_fallback(reference, monkeypatch):
    module = importlib.import_module("crimm.Modeller.MembraneOrienter")
    module._native_function.cache_clear()
    monkeypatch.setattr(module, "find_spec", lambda name: None)
    before = _model(reference)
    coordinates = _coords(before)
    try:
        with pytest.raises(RuntimeError, match="native PPM3 library is unavailable"):
            MembraneOrienter(before).orient()
    finally:
        module._native_function.cache_clear()
    np.testing.assert_array_equal(_coords(before), coordinates)


@pytest.mark.parametrize("failure,expected_status", [
    ("library", 2), ("typing", 3), ("dense", 4), ("count", 1),
])
def test_native_failures_return_without_terminating_python(reference, failure, expected_status):
    script = f"""
import ctypes
import importlib
import runpy
import numpy as np
from pathlib import Path
checks = runpy.run_path({str(Path(__file__).resolve())!r})
model = checks['_model'](Path({str(reference)!r}))
module = importlib.import_module('crimm.Modeller.MembraneOrienter')
coords, names, residues, chains, ids = module.MembraneOrienter(model)._protein_input()
path = str(Path({str(ROOT)!r}) / 'crimm/Modeller/_ppm3/res.lib').encode()
count = len(coords)
if {failure!r} == 'library':
    path += b'.absent'
elif {failure!r} == 'typing':
    names = b'QQQQ' + names[4:]
elif {failure!r} == 'dense':
    coords.fill(0)
else:
    count = 150001
rotation = np.empty((3, 3), dtype=np.float64, order='F')
translation = np.empty(3, dtype=np.float64)
metrics = np.empty(6, dtype=np.float64)
matching = np.empty(3, dtype=np.float64)
status = ctypes.c_int()
module._native_function()(count, coords, names, residues, chains, ids, path,
                          len(path), 7.5, 293.0, 0, b"OPC", rotation, translation,
                          metrics, matching, ctypes.byref(status))
print('NATIVE_RETURNED', status.value)
"""
    result = _run([sys.executable, "-c", script], cwd=ROOT, timeout=120)
    assert f"NATIVE_RETURNED {expected_status}" in result.stdout
