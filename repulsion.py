import numpy as np


class Repulsion:
    """Simple pairwise soft repulsion for the baseline single-room model."""

    def __init__(self, cfg):
        self.cfg = cfg

    def simple_agent_repulsion(self, pos: np.ndarray, active: np.ndarray) -> np.ndarray:
        forces = np.zeros_like(pos)
        idx = active
        if len(idx) < 2:
            return forces

        active_pos = pos[idx]
        dx = active_pos[:, None, 0] - active_pos[None, :, 0]
        dy = active_pos[:, None, 1] - active_pos[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.maximum(dist, 1e-6)
        np.fill_diagonal(dist, np.inf)

        overlap = np.clip(2.0 * self.cfg.radius - dist, 0.0, None)
        ux = dx / dist
        uy = dy / dist

        fx = self.cfg.k_simple_rep * np.sum(overlap * ux, axis=1)
        fy = self.cfg.k_simple_rep * np.sum(overlap * uy, axis=1)

        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces
