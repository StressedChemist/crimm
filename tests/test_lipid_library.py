import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from crimm.Modeller.MembraneBuilder import LipidConformerLibrary, MembraneBuilder, MembraneSpec
from crimm.StructEntities.Model import Model


def test_packaged_library_is_the_default():
    builder = MembraneBuilder(
        Model(1), MembraneSpec(box_xy=(40, 40), box_z=80, upper_leaflet={}, lower_leaflet={})
    )
    conformer = builder.lipid_library.random_conformer("POPC", np.random.default_rng(1))
    assert "P" in conformer
    assert np.isfinite(list(conformer.values())).all()
    assert builder.lipid_library.archive_path.name == "lipid_conformers.npz"
    assert set(builder.lipid_library._coords_by_resname) == {"CHL1", "DMPC", "DOPC", "DPPC", "DSPC", "POPC"}
    for resname, coords in builder.lipid_library._coords_by_resname.items():
        assert len(coords) == 2000
        definitions, _ = builder.topology.load_residue_definitions("Sterol" if resname == "CHL1" else "Lipid")
        residue = definitions[resname].create_residue()
        assert set(builder.lipid_library._atoms_by_resname[resname]) == {atom.name for atom in residue.get_atoms()}
        assert "OH2" not in builder.lipid_library._atoms_by_resname[resname]


def test_compact_library_loads_once(tmp_path, monkeypatch):
    path = tmp_path / "lipids.npz"
    coords = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=float)
    np.savez_compressed(path, POPC_atoms=np.array(["P", "C1"]), POPC_coords=coords)
    original_load = np.load
    loads = []

    def load(*args, **kwargs):
        loads.append(args)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    library = LipidConformerLibrary(path)
    rng = np.random.default_rng(1)
    for _ in range(5):
        conformer = library.random_conformer("popc", rng)
        assert any(np.array_equal(list(conformer.values()), expected) for expected in coords)
    assert len(loads) == 1
    first = library.random_conformer("POPC", np.random.default_rng(1))
    expected = first["P"].copy()
    first["P"][:] = 999
    np.testing.assert_array_equal(
        library.random_conformer("POPC", np.random.default_rng(1))["P"], expected
    )
    with pytest.raises(ValueError, match="No lipid conformers"):
        library.random_conformer("UNKNOWN", rng)


def test_legacy_archive_loads_once(tmp_path, monkeypatch):
    path = tmp_path / "lipids.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for index in range(3):
            rows = [f"1 1 POPC P {index}.0 2.0 3.0 MEMB 1 0.0", "2 1 POPC C1 4.0 5.0 6.0 MEMB 1 0.0"]
            if index == 1:
                rows.reverse()
            if index == 2:
                rows.extend(["3 2 TIP3 OH2 8.0 9.0 10.0 WAT 2 0.0", "4 3 TIP3 OH2 11.0 12.0 13.0 WAT 3 0.0"])
            data = (f"* example\n{len(rows)}\n" + "\n".join(rows) + "\n").encode()
            member = tarfile.TarInfo(f"lipid_lib/popc/conf1/popc_{index}.crd")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    original_open = tarfile.open
    opens = []

    def open_archive(*args, **kwargs):
        opens.append(args)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", open_archive)
    library = LipidConformerLibrary(path)
    rng = np.random.default_rng(2)
    for _ in range(5):
        conformer = library.random_conformer("POPC", rng)
        assert conformer["P"][0] in (0, 1, 2)
        assert set(conformer) == {"P", "C1"}
        np.testing.assert_array_equal(conformer["C1"], [4, 5, 6])
    assert len(opens) == 1


@pytest.mark.parametrize("coords", [np.empty((0, 2, 3)), np.zeros((1, 1, 3)), np.full((1, 2, 3), np.nan)])
def test_invalid_compact_coordinates_fail(tmp_path, coords):
    path = tmp_path / "lipids.npz"
    np.savez_compressed(path, POPC_atoms=np.array(["P", "C1"]), POPC_coords=coords)
    with pytest.raises(ValueError, match="coordinates"):
        LipidConformerLibrary(path)


def test_converter_keeps_all_conformers_and_coordinate_precision(tmp_path):
    from scripts.pack_lipid_library import pack_lipid_library

    source = tmp_path / "lipids.tar.gz"
    destination = tmp_path / "lipids.npz"
    coordinates = {"CHL1": [1.23456, 2.34567], "POPC": [1.2345678901, 2.3456789012], "DMPC": [1.234567890123456, 2.0]}
    with tarfile.open(source, "w:gz") as archive:
        for resname, values in coordinates.items():
            for index, value in enumerate(values):
                data = f"* example\n1\n1 1 {resname} C1 {value:.16g} 0.0 0.0 MEMB 1 0.0\n".encode()
                member = tarfile.TarInfo(f"lipid_lib/{resname.lower()}/conf1/{index}.crd")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    pack_lipid_library(source, destination)
    library = LipidConformerLibrary(destination)
    for resname, expected in coordinates.items():
        np.testing.assert_array_equal(library._coords_by_resname[resname][:, 0, 0], expected)
    with np.load(destination, allow_pickle=False) as data:
        assert data["CHL1_coords"].dtype == np.int32
        assert data["POPC_coords"].dtype == np.int64
        assert data["DMPC_coords"].dtype == np.float64
        assert len(data["POPC_members"]) == 2
        assert len(str(data["_source_sha256"])) == 64


def test_duplicate_lipid_atom_names_fail():
    with pytest.raises(ValueError, match="Duplicate lipid atom names"):
        LipidConformerLibrary._read_crd(
            "* example\n2\n1 1 POPC P 0 0 0 MEMB 1 0\n2 1 POPC P 1 1 1 MEMB 1 0\n", "POPC"
        )


@pytest.mark.slow
def test_every_packaged_conformer_matches_original_archive():
    data_dir = Path(__file__).resolve().parents[1] / "crimm" / "Data"
    source = data_dir / "lipid_lib.tar.gz"
    if not source.is_file():
        pytest.skip("Original archive is optional and is not installed with the package")
    counts = {}
    with np.load(data_dir / "lipid_conformers.npz", allow_pickle=False) as data:
        decoded = {
            key.removesuffix("_coords"): data[key].astype(float) / data[key.removesuffix("_coords") + "_scale"]
            for key in data.files if key.endswith("_coords")
        }
        with tarfile.open(source, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".crd"):
                    continue
                resname = member.name.split("/")[1].upper()
                with archive.extractfile(member) as handle:
                    fields = [line.split() for line in handle.read().decode().splitlines()]
                by_name = {
                    row[3]: [float(value) for value in row[4:7]]
                    for row in fields if len(row) >= 10 and row[2].upper() == resname
                }
                index = counts.get(resname, 0)
                assert data[f"{resname}_members"][index] == member.name
                assert set(data[f"{resname}_atoms"]) == set(by_name)
                np.testing.assert_array_equal(decoded[resname][index], [by_name[name] for name in data[f"{resname}_atoms"]])
                counts[resname] = index + 1
        assert set(counts) == set(decoded)
        assert all(counts[resname] == len(coords) == 2000 for resname, coords in decoded.items())
