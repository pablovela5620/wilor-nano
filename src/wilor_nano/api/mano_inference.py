from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from simplecv.rerun_log_utils import RerunTyroConfig

from wilor_nano.mano_pytorch_simple import ManoSimpleLayer


@dataclass
class ManoConfig:
    rr_config: RerunTyroConfig


def main(config: ManoConfig):
    start_time: float = timer()
    print(f"Start time: {start_time}")
    print(f"rr_config: {config.rr_config}")

    # Initialize models
    pytorch_mano = ManoSimpleLayer(mano_root=Path("/home/pablo/0Dev/repos/pi0-lerobot/data/mano_clean"))

    # Warm-up phase
    print("\n--- Warm-up Phase ---")
    warmup_batch_size = 1
    warmup_pose_coeffs = np.random.rand(warmup_batch_size, 48).astype(np.float32)
    warmup_shape_coeffs = np.random.rand(warmup_batch_size, 10).astype(np.float32)
    warmup_trans = np.random.rand(warmup_batch_size, 3).astype(np.float32)

    # PyTorch warm-up
    _ = pytorch_mano(
        torch.from_numpy(warmup_pose_coeffs), torch.from_numpy(warmup_shape_coeffs), torch.from_numpy(warmup_trans)
    )
    print("PyTorch model warmed up.")

    # Inference phase
    batch_size = 10000
    print(f"\n--- Inference Phase (batch_size={batch_size}) ---")
    pose_coeffs = np.random.rand(batch_size, 48).astype(np.float32)
    shape_coeffs = np.random.rand(batch_size, 10).astype(np.float32)
    trans = np.random.rand(batch_size, 3).astype(np.float32)

    # PyTorch inference
    torch_pose_coeffs = torch.from_numpy(pose_coeffs)
    torch_shape_coeffs = torch.from_numpy(shape_coeffs)
    torch_trans = torch.from_numpy(trans)

    torch_start_time = timer()
    out_torch = pytorch_mano(torch_pose_coeffs, torch_shape_coeffs, torch_trans)
    torch_end_time = timer()
    print(f"PyTorch inference time: {torch_end_time - torch_start_time:.4f} seconds")

    # Compare vertices
    verts_torch_np = out_torch[0].detach().cpu().numpy()

    # Compare joints
    joints_torch_np = out_torch[1].detach().cpu().numpy()

    print("\n--- Summary ---")
    end_time: float = timer()
    print(f"Total time: {end_time - start_time:.2f} seconds")
