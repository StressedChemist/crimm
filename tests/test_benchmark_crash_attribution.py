import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark_pipeline as benchmark


def _stopped_worker(pdb_id, failure_mode, run_charmm, result_path):
    result = benchmark.Result(
        pdb_id=pdb_id, fetch="ok", organize="ok", topo_gen="running", n_protein=1
    )
    with open(result_path, "w") as handle:
        json.dump(result.as_dict(), handle)
    if failure_mode == "timeout":
        benchmark.time.sleep(30)
    raise SystemExit(0 if failure_mode == "exit0" else 3)


class TestBenchmarkCrashAttribution(unittest.TestCase):
    def test_dispatch_preserves_skips_after_worker_exit_and_timeout(self):
        for mode, status in (
            ("exit", "crash"), ("exit0", "crash"), ("timeout", "timeout")
        ):
            with self.subTest(mode=mode), patch.object(
                benchmark, "_run_stages", _stopped_worker
            ):
                result = benchmark._dispatch_worker("TEST", mode, False, timeout=3)
            self.assertEqual(result.topo_gen, status)
            self.assertTrue(result.topo_gen_err)
            self.assertEqual(result.loop_build, benchmark.STATUS_SKIP)
            self.assertEqual(result.charmm_load, benchmark.STATUS_SKIP)

    def test_failure_marks_only_the_running_stage(self):
        for status in (benchmark.STATUS_CRASH, benchmark.STATUS_TIMEOUT):
            for stage in benchmark.STAGES:
                with self.subTest(status=status, stage=stage):
                    result = benchmark.Result(pdb_id="TEST", n_protein=1)
                    setattr(result, stage, "running")
                    benchmark._annotate_subprocess_failure(result, status, "stopped")
                    self.assertEqual(getattr(result, stage), status)
                    self.assertEqual(getattr(result, stage + "_err"), "stopped")
                    for other in benchmark.STAGES:
                        if other != stage:
                            self.assertEqual(getattr(result, other), benchmark.STATUS_SKIP)

    def test_completed_and_skipped_stages_remain_unchanged(self):
        result = benchmark.Result(
            pdb_id="TEST", fetch="ok", organize="ok", topo_gen="ok"
        )
        before = result.as_dict()
        benchmark._annotate_subprocess_failure(result, "crash", "stopped")
        self.assertEqual(result.as_dict(), before)

    def test_worker_checkpoints_each_stage_before_native_failure(self):
        cases = [(stage, True) for stage in benchmark.STAGES]
        cases.extend((stage, False) for stage in ("topo_gen", "charmm_load", None))
        cases.append(("topology_constructor", True))
        for stage, has_protein in cases:
            with self.subTest(stage=stage, has_protein=has_protein), \
                    tempfile.TemporaryDirectory() as directory:
                checkpoint = str(Path(directory) / "result.json")
                chain = SimpleNamespace(is_continuous=lambda: False)
                model = SimpleNamespace(
                    protein=[chain] if has_protein else [],
                    RNA=[] if has_protein else [object()], DNA=[]
                )
                actions = {
                    name: Mock() for name in benchmark.STAGES + ("topology_constructor",)
                }
                actions["fetch"].return_value = object()
                actions["organize"].return_value = model
                actions["topology_constructor"].return_value = SimpleNamespace(
                    generate_model=actions["topo_gen"]
                )
                if stage is not None:
                    actions[stage].side_effect = SystemExit("native failure")
                expected_stage = "topo_gen" if stage == "topology_constructor" else stage
                modules = {
                    "crimm.Fetchers": SimpleNamespace(fetch_rcsb=actions["fetch"]),
                    "crimm.StructEntities.OrganizedModel": SimpleNamespace(
                        OrganizedModel=actions["organize"]
                    ),
                    "crimm.Modeller": SimpleNamespace(
                        TopologyGenerator=actions["topology_constructor"]
                    ),
                    "crimm.Modeller.LoopBuilder": SimpleNamespace(
                        ChainLoopBuilder=lambda *args, **kwargs: SimpleNamespace(
                            build_from_alphafold=actions["loop_build"]
                        )
                    ),
                    "crimm.Adaptors.pyCHARMMAdaptors": SimpleNamespace(
                        load_model=actions["charmm_load"], empty_charmm=Mock()
                    ),
                }
                with patch.dict(sys.modules, modules):
                    if stage is None:
                        benchmark._run_stages("TEST", None, False, checkpoint)
                    else:
                        with self.assertRaises(SystemExit):
                            benchmark._run_stages("TEST", None, True, checkpoint)
                with open(checkpoint) as handle:
                    data = json.load(handle)
                if not has_protein:
                    self.assertEqual(data["loop_build"], benchmark.STATUS_SKIP)
                if stage is None:
                    self.assertEqual(data["charmm_load"], benchmark.STATUS_SKIP)
                else:
                    self.assertEqual(data[expected_stage], "running")
                self.assertEqual(
                    [name for name in benchmark.STAGES if data[name] == "running"],
                    [expected_stage] if stage is not None else [],
                )


if __name__ == "__main__":
    unittest.main()
