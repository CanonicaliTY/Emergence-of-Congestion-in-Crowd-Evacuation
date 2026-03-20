from dataclasses import dataclass, field
from repulsion import Repulsion
import numpy as np


@dataclass
class Config:
    # Geometry
    room_w: float = 20.0
    room_h: float = 20.0
    door_w: float = 2.0
    corridor_len: float = 8.0   # used as simplified staircase bottleneck
    exits: list = field(default_factory=lambda: [
        {"x": 10.0, "y": 8.0, "w": 2.0, "side" : "bottom"},
        {"x":20.0, "y":21.0, "w":2.0, "side" : "right"}
    ])
    slope_angle: float = 12.0

    # Agents
    n_agents: int = 150
    radius: float = 0.22
    desired_speed: float = 1.6
    corridor_speed_factor: float = 0.8
    tau: float = 0.5

    # Forces
    k_simple_rep: float = 20.0
    k_coulomb_rep: float = 2.0
    k_yukawa_rep: float = 2.0
    k_wall: float = 100.0

    # Time integration
    dt: float = 0.05
    t_max: float = 240.0
    speed_cap: float = 3.0

    # Output
    snapshot_every: int = 10
    seed: int = 42


class Simulation:
    """2D social-force-style evacuation simulation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.room_top = cfg.corridor_len + cfg.room_h

        self.pos = self._initialize_agents()
        self.vel = np.zeros_like(self.pos)
        self.active = np.ones(cfg.n_agents, dtype=bool)

        self.times = [0.0]
        self.remaining = [cfg.n_agents]
        self.snapshots = [self.pos[self.active].copy()]

    def _active_indices(self) -> np.ndarray:
        return np.where(self.active)[0]

    def _initialize_agents(self) -> np.ndarray:
        """Randomly place agents inside the lecture theatre, avoiding large overlaps."""
        rng = np.random.default_rng(self.cfg.seed)
        pos = np.zeros((self.cfg.n_agents, 2))

        count = 0
        attempts = 0
        max_attempts = 200000

        while count < self.cfg.n_agents and attempts < max_attempts:
            candidate = np.array(
                [
                    rng.uniform(self.cfg.radius, self.cfg.room_w - self.cfg.radius),
                    rng.uniform(
                        self.cfg.corridor_len + self.cfg.radius,
                        self.room_top - self.cfg.radius,
                    ),
                ]
            )

            if count == 0:
                pos[count] = candidate
                count += 1
            else:
                distances = np.linalg.norm(pos[:count] - candidate, axis=1)
                if np.all(distances > 2.1 * self.cfg.radius):
                    pos[count] = candidate
                    count += 1

            attempts += 1

        if count < self.cfg.n_agents:
            raise RuntimeError("Could not place all agents without overlap. Try fewer agents.")

        return pos

    def _targets(self) -> np.ndarray:
        """
        Calculates the nearest exit for each active agent (vectorized).
        Adds a small noise to distances to encourage more natural distribution.
        """
        targets = np.zeros((self.cfg.n_agents, 2))
        idx = self._active_indices()
        if len(idx) == 0:
            return targets

        pos = self.pos[idx]
        
        # 1. Pre-calculate exit centers and offset "pull" points
        exit_centers = []
        offset_targets = []
        for ex in self.cfg.exits:
            center = np.array([ex['x'], ex['y']])
            exit_centers.append(center)
            
            offset = 1.0  # Point 1 meter past the door to ensure crossing
            if ex["side"] == "bottom": t = center + [0, -offset]
            elif ex["side"] == "top": t = center + [0, offset]
            elif ex["side"] == "left": t = center + [-offset, 0]
            elif ex["side"] == "right": t = center + [offset, 0]
            else: t = center
            offset_targets.append(t)
        
        exit_centers = np.array(exit_centers)
        offset_targets = np.array(offset_targets)

        # 2. Calculate distances from every agent to every exit center
        # pos: (N_active, 2), exit_centers: (N_exits, 2)
        # Using broadcasting for shape: (N_active, N_exits, 2)
        diffs = pos[:, np.newaxis, :] - exit_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        # 3. Add small noise (0.5m) to the distance metrics
        # This prevents everyone on a perfect symmetry line from picking the same exit.
        rng = np.random.default_rng(self.cfg.seed + int(self.times[-1] * 10))
        dists += rng.normal(0, 0.3, size=dists.shape)

        # 4. Assign best target to each agent
        best_exit_indices = np.argmin(dists, axis=1)
        targets[idx] = offset_targets[best_exit_indices]

        return targets

    def _goal_force(self) -> np.ndarray:
        forces = np.zeros_like(self.pos)

        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        targets = self._targets()[idx]
        pos = self.pos[idx]
        vel = self.vel[idx]

        directions = targets - pos
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        directions = directions / norms

        desired_speed = np.full((len(idx), 1), self.cfg.desired_speed)
        in_corridor = pos[:, 1] < self.cfg.corridor_len
        desired_speed[in_corridor] *= self.cfg.corridor_speed_factor

        desired_vel = desired_speed * directions
        forces[idx] = (desired_vel - vel) / self.cfg.tau
        return forces

    def _agent_repulsion(self) -> np.ndarray:
        """
        Simple soft repulsion when agents overlap.
        This is intentionally minimal for the first commit.
        """
        return Repulsion(self.cfg).yukawa_like_agent_repulsion(self.pos, self._active_indices())

    def _wall_force(self) -> np.ndarray:
        """
        Generalized wall forces that account for multiple exits on any side.
        """
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        pos = self.pos[idx]
        x = pos[:, 0]
        y = pos[:, 1]

        fx = np.zeros(len(idx))
        fy = np.zeros(len(idx))

        # Helper to check if an agent is in any exit gap on a specific side
        def in_any_gap(side, coord):
            gap_mask = np.zeros(len(idx), dtype=bool)
            for ex in self.cfg.exits:
                if ex["side"] == side:
                    # coord is x if side top/bottom, y if side left/right
                    in_range = (coord > ex["x" if side in ["bottom", "top"] else "y"] - ex["w"]/2) & \
                               (coord < ex["x" if side in ["bottom", "top"] else "y"] + ex["w"]/2)
                    gap_mask |= in_range
            return gap_mask

        # Left wall (x=0)
        near_left = x < self.cfg.radius
        fx += self.cfg.k_wall * (self.cfg.radius - x) * near_left * (~in_any_gap("left", y))

        # Right wall (x=room_w)
        near_right = x > (self.cfg.room_w - self.cfg.radius)
        fx -= self.cfg.k_wall * (x - (self.cfg.room_w - self.cfg.radius)) * near_right * (~in_any_gap("right", y))

        # Top wall (y=room_top)
        near_top = y > (self.room_top - self.cfg.radius)
        fy -= self.cfg.k_wall * (y - (self.room_top - self.cfg.radius)) * near_top * (~in_any_gap("top", y))

        # Bottom wall (y=corridor_len)
        near_bottom = (y < self.cfg.corridor_len + self.cfg.radius) & (y > self.cfg.corridor_len - self.cfg.radius)
        fy += self.cfg.k_wall * ((self.cfg.corridor_len + self.cfg.radius) - y) * near_bottom * (~in_any_gap("bottom", x))

        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces
    def slope_force(self) -> np.ndarray:
        """
        Force that simulates the spatial height difference (gravity-like pull downhill).
        Assumes the room slopes upwards from y = corridor_len to y = room_top.
        """
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) == 0:
            return forces
        
        pos = self.pos[idx]
        y = pos[:, 1]
        
        # Only agents inside the room (y >= corridor_len) feel the slope
        # Stage is at y=corridor_len, seats slope up from there.
        in_room = (y >= self.cfg.corridor_len)
        
        # Downhill direction is (0, -1)
        # Using a simple k_slope factor. Here we use tan(angle) as the magnitude.
        # F = -k * dy/dz = (0, -k * tan(theta))
        k_gravity = 5.0 # Reduced from 15.0 to prevent overpowering the goal force
        fy_slope = -k_gravity * np.tan(np.deg2rad(self.cfg.slope_angle))
        
        forces[idx[in_room], 1] = fy_slope
        return forces
        

    def _limit_speed(self) -> None:
        idx = self._active_indices()
        if len(idx) == 0:
            return

        speeds = np.linalg.norm(self.vel[idx], axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-8)
        factors = np.minimum(1.0, self.cfg.speed_cap / speeds)
        self.vel[idx] *= factors

    def _remove_exited_agents(self) -> None:
        """Remove agents who have passed through any of the defined exits."""
        for ex in self.cfg.exits:
            if ex["side"] == "bottom":
                exited = self.active & (self.pos[:, 1] < ex["y"] - self.cfg.radius)
            elif ex["side"] == "top":
                exited = self.active & (self.pos[:, 1] > ex["y"] + self.cfg.radius)
            elif ex["side"] == "left":
                exited = self.active & (self.pos[:, 0] < ex["x"] - self.cfg.radius)
            elif ex["side"] == "right":
                exited = self.active & (self.pos[:, 0] > ex["x"] + self.cfg.radius)
            
            self.active[exited] = False
            self.vel[~self.active] = 0.0

    def step(self) -> None:
        total_force = (
            self._goal_force()
            + self._agent_repulsion()
            + self._wall_force()
            + self.slope_force()
        )

        idx = self._active_indices()
        if len(idx) == 0:
            return

        self.vel[idx] += total_force[idx] * self.cfg.dt
        self._limit_speed()
        self.pos[idx] += self.vel[idx] * self.cfg.dt
        self._remove_exited_agents()

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        n_steps = int(self.cfg.t_max / self.cfg.dt)

        for step in range(1, n_steps + 1):
            if not np.any(self.active):
                break

            self.step()

            current_time = step * self.cfg.dt
            n_remaining = int(np.sum(self.active))

            self.times.append(current_time)
            self.remaining.append(n_remaining)

            if step % self.cfg.snapshot_every == 0:
                self.snapshots.append(self.pos[self.active].copy())

        return np.array(self.times), np.array(self.remaining)
