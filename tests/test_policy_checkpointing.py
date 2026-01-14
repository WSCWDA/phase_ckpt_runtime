import tempfile
import unittest

import torch

from phase_runtime import PhaseAwareCheckpointRuntime, PhaseRuntimeConfig, PhaseState
from policy_controller import CheckpointPolicyConfig, CheckpointPolicyController


class PolicyCheckpointingTests(unittest.TestCase):
    def _make_runtime(self, output_dir: str) -> PhaseAwareCheckpointRuntime:
        runtime_cfg = PhaseRuntimeConfig(
            strategy="sync",
            ckpt_interval_steps=1,
            async_queue_size=1,
            async_timeout_s=0.1,
            total_steps=4,
        )
        policy_cfg = CheckpointPolicyConfig(
            base_interval=1,
            max_staleness_steps=10,
            min_interval_steps=0,
            high_latency_s=10.0,
            force_sync_on_staleness=True,
            compression_level_low=1,
            compression_level_high=3,
        )
        controller = CheckpointPolicyController(policy_cfg)
        return PhaseAwareCheckpointRuntime(runtime_cfg, output_dir, policy_controller=controller)

    def test_delta_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._make_runtime(tmpdir)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            phase_state = PhaseState(False, False, False, {}, "A0_D0_C0")

            runtime.maybe_checkpoint(
                1,
                model,
                optimizer,
                {"step_time": 0.1},
                phase_state=phase_state,
                observation_stats={"staleness_steps": 0, "ckpt_latency": None},
            )

            for param in model.parameters():
                param.data.add_(1.0)

            phase_state_delta = PhaseState(False, True, False, {}, "A0_D1_C0")
            runtime.maybe_checkpoint(
                2,
                model,
                optimizer,
                {"step_time": 0.1},
                phase_state=phase_state_delta,
                observation_stats={"staleness_steps": 0, "ckpt_latency": None},
            )

            payload = torch.load(runtime._checkpoint_path(2))
            self.assertEqual(payload["checkpoint_type"], "delta")
            self.assertEqual(payload["base_step"], 1)
            weight_delta = payload["model_delta"]["weight"]
            self.assertTrue(torch.allclose(weight_delta, torch.ones_like(weight_delta)))
            runtime.close()

    def test_compression_checkpoint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._make_runtime(tmpdir)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            phase_state = PhaseState(False, False, True, {}, "A0_D0_C1")

            runtime.maybe_checkpoint(
                1,
                model,
                optimizer,
                {"step_time": 0.1},
                phase_state=phase_state,
                observation_stats={"staleness_steps": 0, "ckpt_latency": None},
            )

            payload = torch.load(runtime._checkpoint_path(1))
            self.assertEqual(payload["checkpoint_type"], "compressed")
            self.assertEqual(payload["compression_level"], 3)
            self.assertIsInstance(payload["compressed_payload"], (bytes, bytearray))
            runtime.close()


if __name__ == "__main__":
    unittest.main()
