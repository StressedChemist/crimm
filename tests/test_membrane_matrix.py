import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.membrane_matrix import CASES, SEEDS, run_case


def test_matrix_has_all_18_wet_cases():
    assert len(CASES) == 6 and SEEDS == (12345,12346,12347)
    assert {row[0] for row in CASES} == {"popc-control", "popc-helix", "six-control",
        "asymmetric-control", "asymmetric-helix-in", "asymmetric-helix-out"}
    assert CASES[-1][2] == "out" and CASES[-2][2] == "in"
    assert all(sum(leaf.values()) == 100 for row in CASES[2:] for leaf in row[3:])


def test_matrix_does_not_accept_a_failed_child(tmp_path):
    output = tmp_path / "popc-control-12345"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"case_acceptance_passed": True, "stage": "minimization"}))
    with patch("scripts.membrane_matrix.subprocess.run", return_value=SimpleNamespace(returncode=2)) as run:
        record = run_case(tmp_path, Path("/unused"), CASES[0], 12345)
    assert record["passed"] is False and record["stage"] == "minimization"
    assert "--dry" not in run.call_args.args[0]


def test_preset_matrix_selects_matching_native_code_and_pc_temperature(tmp_path):
    for preset, code in (("ERm", "ERm"), ("DMPC", "MPC")):
        case = (f"oprlm-{preset}", "control", "in", None, None)
        with patch("scripts.membrane_matrix.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run:
            result = run_case(tmp_path, Path("/unused"), case, 12345, preset)
        command = run.call_args.args[0]
        assert command[command.index("--membrane") + 1] == preset
        assert command[command.index("--ppm-membrane") + 1] == code
        assert ("--cholesterol-percent" in command) == (preset == "DMPC")
        if preset == "DMPC":
            assert command[command.index("--temperature") + 1] == "333.15"
        assert not result["passed"]
