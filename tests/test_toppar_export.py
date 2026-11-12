"""Parameter exports preserve values and declare shared NBFIX types."""

from pathlib import Path

from crimm.IO.PRMParser import categorize_lines, parse_line_dict
from crimm.Modeller.TopoLoader import TopologyGenerator, toppar_dir


def test_parameter_export_preserves_original_title(tmp_path):
    topology = TopologyGenerator()
    for chain_type in ("Polypeptide(L)", "Lipid", "Sterol"):
        topology.load_residue_definitions(chain_type)
    topology.write_toppar(path=str(tmp_path))
    assert not (tmp_path / "toppar/lipid_cholesterol.prm").exists()

    for name in ("protein", "lipid"):
        original = "\n".join(topology.param_dict[name]._raw_data_strings)
        exported = (tmp_path / "toppar" / f"{name}.prm").read_text()
        assert exported.startswith(f"* {name.upper()} PRM exported by crimm\n*\n")
        assert parse_line_dict(categorize_lines(exported.splitlines())) == topology.param_dict[name].param_dict
        if name == "lipid":
            declarations = (tmp_path / "toppar/membrane_types.prm").read_text()
            mass_names = {line.split()[2] for line in declarations.splitlines() if line.startswith("MASS")}
            assert all(set(pair) <= mass_names for pair in topology.param_dict[name].param_dict["nbfix"])
            assert "HGA1" in topology.param_dict[name].param_dict["nonbonded"]

    water_stream = (tmp_path / "toppar/water_ions.str").read_text()
    original = (Path(toppar_dir) / "water_ions.str").read_text()
    original_rtf, original_prm = original.split("read para card flex @app")
    exported_rtf, exported_prm = water_stream.split("read para card flex @app")
    assert original_rtf == exported_rtf
    assert exported_prm == "".join(line for line in original_prm.splitlines(keepends=True)
                                 if not line.strip().upper().startswith(("ATOMS", "MASS")))
