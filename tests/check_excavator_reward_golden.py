from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_import_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pact_dir = Path(__file__).resolve().parents[1]
    for p in (str(project_root), str(pact_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> None:
    _ensure_import_paths()

    from PACT.simulation.rewards import ExcavatorPoseReward

    replay_path = Path(__file__).resolve().parent / "golden_excavator_replay_qpos.npy"
    ee_path = Path(__file__).resolve().parent / "golden_excavator_ee_qpos.npy"
    replay_qpos = np.load(replay_path)
    ee_qpos = np.load(ee_path)

    replay_r = ExcavatorPoseReward().evaluate(replay_qpos)
    ee_r = ExcavatorPoseReward().evaluate(ee_qpos[1:])

    print(f"replay_path: {replay_path}")
    print(f"replay qpos shape: {replay_qpos.shape}")
    print(f"replay reward: {replay_r.reward}")
    print(f"replay flags: {replay_r.phase_flags}")
    print(f"replay switch_steps: {replay_r.phase_switch_steps}")
    print(f"ee_path: {ee_path}")
    print(f"ee qpos shape: {ee_qpos.shape}")
    print(f"ee reward: {ee_r.reward}")
    print(f"ee flags: {ee_r.phase_flags}")
    print(f"ee switch_steps: {ee_r.phase_switch_steps}")

    # Replay reward is the authoritative one for collection success / HDF5 observations.
    assert replay_r.reward == 4.0, f"expected replay reward 4.0, got {replay_r.reward}"
    assert replay_r.phase_flags.get("dump_open", False), "expected replay dump_open=True"
    assert replay_r.phase_flags.get("swing_to_target", False), "expected replay swing_to_target=True"
    # EE trajectory may be 4.0 or 2.8 depending on policy/IK; replay reaching 4.0 is the fix target.
    if ee_r.phase_flags.get("dump_open", False):
        assert ee_r.reward == 4.0, f"expected EE reward 4.0 when dump_open, got {ee_r.reward}"


if __name__ == "__main__":
    main()

