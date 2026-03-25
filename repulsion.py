import numpy as np

class Repulsion:
  def __init__(self, cfg):
    self.cfg = cfg
        
  def simple_agent_repulsion(self, pos, active, dx=None, dy=None, dist=None, radii=None, vel=None):
    forces = np.zeros_like(pos)
    idx = active
    if len(idx) < 2:
      return forces
        
    pos = pos[idx]
    if dx is None or dy is None or dist is None:
      dx = pos[:, None, 0] - pos[None, :, 0]
      dy = pos[:, None, 1] - pos[None, :, 1]
      dist = np.sqrt(dx**2 + dy**2)
      dist = np.maximum(dist, 1e-6)
      np.fill_diagonal(dist, np.inf)

    if radii is not None:
        two_r = radii[:, None] + radii[None, :]
    else:
        two_r = 2.0 * self.cfg.radius
    overlap = np.clip(two_r - dist, 0.0, None)
    ux = dx / dist
    uy = dy / dist

    f_rep = self.cfg.k_simple_rep * overlap
    
    # Normal damping term
    if vel is not None and hasattr(self.cfg, 'gamma_damp'):
        dvx = vel[:, None, 0] - vel[None, :, 0]
        dvy = vel[:, None, 1] - vel[None, :, 1]
        # Project relative velocity onto normal direction (ux, uy)
        v_normal = dvx * ux + dvy * uy
        f_rep -= self.cfg.gamma_damp * overlap * v_normal

    fx = np.sum(f_rep * ux, axis=1)
    fy = np.sum(f_rep * uy, axis=1)

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

    fx = self.cfg.k_coulomb_rep * np.sum(ux / dist**2, axis=1)
    fy = self.cfg.k_coulomb_rep * np.sum(uy / dist**2, axis=1)

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

    fx = self.cfg.k_yukawa_rep * np.sum(np.exp(-dist) * ux, axis=1)
    fy = self.cfg.k_yukawa_rep * np.sum(np.exp(-dist) * uy, axis=1)

    forces[idx, 0] = fx
    forces[idx, 1] = fy
    return forces

  def helbing_agent_repulsion(self, pos, active, dx=None, dy=None, dist=None, radii=None):
    """
    Helbing et al. (2000) model:
      Social force:  A * exp((2r - d) / B) * n_ij   (long-range, keeps personal space)
      Body force:    k_body * max(0, 2r - d) * n_ij  (contact-only, prevents overlap)
    """
    forces = np.zeros_like(pos)
    idx = active
    if len(idx) < 2:
      return forces

    pos = pos[idx]
    if dx is None or dy is None or dist is None:
      dx = pos[:, None, 0] - pos[None, :, 0]
      dy = pos[:, None, 1] - pos[None, :, 1]
      dist = np.sqrt(dx**2 + dy**2)
      dist = np.maximum(dist, 1e-6)
      np.fill_diagonal(dist, np.inf)

    ux = dx / dist
    uy = dy / dist

    if radii is not None:
        two_r = radii[:, None] + radii[None, :]
    else:
        two_r = 2.0 * self.cfg.radius

    # Social force (always active, exponential decay)
    f_social = self.cfg.A_social * np.exp((two_r - dist) / self.cfg.B_social)

    # Body compression force (contact only)
    overlap = np.clip(two_r - dist, 0.0, None)
    f_body = self.cfg.k_body * overlap

    f_total = f_social + f_body

    fx = np.sum(f_total * ux, axis=1)
    fy = np.sum(f_total * uy, axis=1)

    forces[idx, 0] = fx
    forces[idx, 1] = fy
    return forces