import numpy as np

class Repulsion:
  def __init__(self, cfg):
    self.cfg = cfg
        
  def simple_agent_repulsion(self, pos, active):
    forces = np.zeros_like(pos)
    idx = active
    if len(idx) < 2:
      return forces
        
    pos = pos[idx]
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1e-6)
    np.fill_diagonal(dist, np.inf)

    overlap = np.clip(2.0 * self.cfg.radius - dist, 0.0, None)
    ux = dx / dist
    uy = dy / dist

    fx = self.cfg.k_rep * np.sum(overlap * ux, axis=1)
    fy = self.cfg.k_rep * np.sum(overlap * uy, axis=1)

    forces[idx, 0] = fx
    forces[idx, 1] = fy
    return forces
    
  def coulomb_agent_repulsion(self, pos, active):
    forces = np.zeros_like(pos)
    idx = active
    if len(idx) < 2:
      return forces
        
    pos = pos[idx]
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1e-6)
    np.fill_diagonal(dist, np.inf)

    ux = dx / dist
    uy = dy / dist

    fx = self.cfg.k_rep * np.sum(ux / dist**2, axis=1)
    fy = self.cfg.k_rep * np.sum(uy / dist**2, axis=1)

    forces[idx, 0] = fx
    forces[idx, 1] = fy
    return forces

  def yukawa_like_agent_repulsion(self, pos, active):
    forces = np.zeros_like(pos)
    idx = active
    if len(idx) < 2:
      return forces
        
    pos = pos[idx]
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1e-6)
    np.fill_diagonal(dist, np.inf)

    ux = dx / dist
    uy = dy / dist

    fx = self.cfg.k_rep * np.sum(np.exp(-dist) * ux, axis=1)
    fy = self.cfg.k_rep * np.sum(np.exp(-dist) * uy, axis=1)

    forces[idx, 0] = fx
    forces[idx, 1] = fy
    return forces