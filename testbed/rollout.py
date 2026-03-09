from __future__ import annotations

from typing import List, Tuple

import numpy as np

from PACT.simulation.backends import SimulationBackend
from PACT.testbed.render import close_render, init_render, update_render


def rollout_episode(
    backend: SimulationBackend,
    policy,
    episode_len: int,
    render_cam: str | None = None,
    render_dt: float = 0.02,
) -> Tuple[List[object], List[np.ndarray]]:
    ts = backend.reset()
    episode = [ts]
    actions: List[np.ndarray] = []

    plt_img = None
    if render_cam is not None:
        first_image = backend.render(height=480, width=640, camera_id=render_cam)
        plt_img = init_render(first_image)

    for _ in range(episode_len):
        action = policy(ts) if callable(policy) else policy.act(ts.observation)
        actions.append(action)
        ts = backend.step(action)
        episode.append(ts)
        if plt_img is not None:
            image = backend.render(height=480, width=640, camera_id=render_cam)
            update_render(plt_img, image, pause_time=render_dt)

    if plt_img is not None:
        close_render()

    return episode, actions

