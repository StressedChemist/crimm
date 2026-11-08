"""Run wet, restrained membrane preparations in fresh CHARMM processes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crimm.Modeller.MembranePresets import get_membrane_preset, list_membrane_presets
SYMMETRIC = {"POPC":20, "DOPC":20, "DMPC":15, "DPPC":15, "DSPC":10, "CHL1":20}
UPPER = {"POPC":40, "DOPC":40, "CHL1":20}
LOWER = {"DMPC":25, "DPPC":25, "DSPC":30, "CHL1":20}
CASES = (
    ("popc-control", "control", "in", None, None),
    ("popc-helix", "helix", "in", None, None),
    ("six-control", "control", "in", SYMMETRIC, SYMMETRIC),
    ("asymmetric-control", "control", "in", UPPER, LOWER),
    ("asymmetric-helix-in", "helix", "in", UPPER, LOWER),
    ("asymmetric-helix-out", "helix", "out", UPPER, LOWER),
)
SEEDS = (12345, 12346, 12347)


def run_case(directory, library, case, seed, preset=None):
    name, model, topology, upper, lower = case
    output = directory / f"{name}-{seed}"
    command = [sys.executable, str(ROOT / "scripts/membrane_pycharmm.py"), "--case", model,
               "--topology", topology, "--ppm-membrane", "OPC", "--seed", str(seed),
               "--output", str(output), "--charmm-lib", str(library)]
    if upper is not None:
        command += ["--upper-lipid-ratios", json.dumps(upper), "--lower-lipid-ratios", json.dumps(lower)]
    if preset is not None:
        pc_family = preset in {"POPC", "DOPC", "DSPC", "DMPC", "DPPC"}
        command[command.index("--ppm-membrane") + 1] = get_membrane_preset(
            preset, 20 if pc_family else None)["ppm_membrane"]
        command += ["--membrane", preset]
        if pc_family:
            command += ["--cholesterol-percent", "20", "--temperature", "333.15"]
    start = time.perf_counter()
    with (directory / f"{name}-{seed}.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record_path = output / "result.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    return {"case": name, "seed": seed, "exit_code": process.returncode,
            "passed": process.returncode == 0 and record.get("case_acceptance_passed") is True,
            "stage": record.get("stage", "startup"), "record": str(record_path),
            "elapsed_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charmm-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oprlm", action="store_true", help="Test all 23 presets at three seeds (69 controls).")
    args = parser.parse_args()
    directory = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="crimm-membrane-matrix-"))
    if args.output:
        directory.mkdir(parents=True, exist_ok=False)
    print(directory, flush=True)
    cases = [(case, None) for case in CASES]
    if args.oprlm:
        cases = [((f"oprlm-{name}", "control", "in", None, None), name)
                 for name in list_membrane_presets()]
    results = []
    with ThreadPoolExecutor(max_workers=3) as workers:
        jobs = [workers.submit(run_case, directory, args.charmm_lib.resolve(), case, seed, preset)
                for case, preset in cases for seed in SEEDS]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            results.sort(key=lambda row: (row["case"], row["seed"]))
            (directory / "matrix.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
            print(f"{result['case']} / {result['seed']}: {'passed' if result['passed'] else 'FAILED'}", flush=True)
    if len(results) != len(cases) * len(SEEDS) or not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
