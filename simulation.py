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
    seat_rows: list = field(default_factory=lambda: [
        {"y": 11.0, "w": 0.2},
        {"y": 12.0, "w": 0.2},
        {"y": 13.0, "w": 0.2},
        {"y": 14.0, "w": 0.2},
        {"y": 15.0, "w": 0.2},
        {"y": 16.0, "w": 0.2},
        {"y": 17.0, "w": 0.2},
        {"y": 18.0, "w": 0.2},
        {"y": 19.0, "w": 0.2},
        {"y": 20.0, "w": 0.2},
        {"y": 24.0, "w": 0.2},
        {"y": 25.0, "w": 0.2},
        {"y": 26.0, "w": 0.2}
    ])
    aisles: list = field(default_factory=lambda: [
        {"x": 5.0, "w": 1.2},
        {"x": 15.0, "w": 1.2}
    ])
    slope_angle: float = 12.0

    # Agents
    n_agents: int = 150
    radius: float = 0.22
    desired_speed: float = 2.0
    corridor_speed_factor: float = 0.8
    tau: float = 0.3

    # Forces
    k_simple_rep: float = 20.0
    k_coulomb_rep: float = 2.0
    k_yukawa_rep: float = 2.0
    k_wall: float = 100.0
    k_seat: float = 100.0
    k_fluct: float = 10.0  # New: Jitter to break symmetry

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

        # 5. Hybrid Aisle Targeting (The "Dual Approach")
        # We combine the "Sticky Aisle" rule-based targeting with the 
        # "Potential Gradient" physical force for maximum reliability.
        
        x_active = pos[:, 0]
        y_active = pos[:, 1]
        
        # Define seating zones [10-21] and [23-28]
        in_block_1 = (y_active >= 10.0 - 0.2) & (y_active <= 21.0 + 0.2)
        in_block_2 = (y_active >= 23.0 - 0.2) & (y_active <= 28.0 + 0.2)
        in_seating_zone = in_block_1 | in_block_2
        
        # Exempt agents already in door gaps
        in_any_door = (self._in_any_door_gap("bottom", x_active) | 
                       self._in_any_door_gap("top", x_active) | 
                       self._in_any_door_gap("left", y_active) | 
                       self._in_any_door_gap("right", y_active))
        in_seating_zone &= (~in_any_door)
        
        if np.any(in_seating_zone):
            active_in_aisle = self._in_any_aisle_gap(x_active)
            aisle_xs = np.array([a["x"] for a in self.cfg.aisles])
            
            sz_idx = np.where(in_seating_zone)[0]
            sz_x = x_active[sz_idx][:, np.newaxis]
            dist_to_aisle = np.abs(sz_x - aisle_xs[np.newaxis, :])
            
            # Add noise to split the crowd at the middle
            rng_aisle = np.random.default_rng(self.cfg.seed + int(self.times[-1] * 100))
            dist_to_aisle += rng_aisle.normal(0, 3, size=dist_to_aisle.shape)
            
            best_aisle_idx = np.argmin(dist_to_aisle, axis=1)
            nearest_aisle_x = aisle_xs[best_aisle_idx]
            global_sz_idx = idx[sz_idx]
            
            # Sideways to aisle
            mask_sideways = ~active_in_aisle[sz_idx]
            if np.any(mask_sideways):
                t_idx = global_sz_idx[mask_sideways]
                targets[t_idx, 0] = nearest_aisle_x[mask_sideways]
                targets[t_idx, 1] = y_active[sz_idx[mask_sideways]]

            # Vertically down the aisle (Sticky Mode)
            mask_vertical = active_in_aisle[sz_idx]
            if np.any(mask_vertical):
                v_idx = global_sz_idx[mask_vertical]
                targets[v_idx, 0] = nearest_aisle_x[mask_vertical]

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

    def _in_any_door_gap(self, side: str, coords: np.ndarray) -> np.ndarray:
        """Vectorized check if agents are in any exit gap on a specific side."""
        gap_mask = np.zeros(len(coords), dtype=bool)
        for ex in self.cfg.exits:
            if ex["side"] == side:
                door_coord = ex["x" if side in ["bottom", "top"] else "y"]
                in_range = (coords > door_coord - ex["w"]/2) & (coords < door_coord + ex["w"]/2)
                gap_mask |= in_range
        return gap_mask

    def _in_any_aisle_gap(self, x_coords: np.ndarray) -> np.ndarray:
        """Vectorized check if agents are in any vertical seat-aisle gap."""
        gap_mask = np.zeros(len(x_coords), dtype=bool)
        for aisle in self.cfg.aisles:
            in_range = (x_coords > aisle["x"] - aisle["w"]/2) & (x_coords < aisle["x"] + aisle["w"]/2)
            gap_mask |= in_range
        return gap_mask

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

        # Left wall (x=0)
        near_left = x < self.cfg.radius
        fx += self.cfg.k_wall * (self.cfg.radius - x) * near_left * (~self._in_any_door_gap("left", y))

        # Right wall (x=room_w)
        near_right = x > (self.cfg.room_w - self.cfg.radius)
        fx -= self.cfg.k_wall * (x - (self.cfg.room_w - self.cfg.radius)) * near_right * (~self._in_any_door_gap("right", y))

        # Top wall (y=room_top)
        near_top = y > (self.room_top - self.cfg.radius)
        fy -= self.cfg.k_wall * (y - (self.room_top - self.cfg.radius)) * near_top * (~self._in_any_door_gap("top", y))

        # Bottom wall (y=corridor_len)
        near_bottom = (y < self.cfg.corridor_len + self.cfg.radius) & (y > self.cfg.corridor_len - self.cfg.radius)
        fy += self.cfg.k_wall * ((self.cfg.corridor_len + self.cfg.radius) - y) * near_bottom * (~self._in_any_door_gap("bottom", x))

        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces

    def _fluctuation_force(self) -> np.ndarray:
        """Adds random Brownian-like jitter to break unphysical symmetry."""
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        rng = np.random.default_rng(self.cfg.seed + int(self.times[-1] * 1000))
        # Large enough kick to break stalemates but small enough not to look jittery
        forces[idx] = rng.normal(0, self.cfg.k_fluct, size=(len(idx), 2))
        return forces

    def seat_force(self) -> np.ndarray:
        """
        Spatially-varying seat potential.
        Potential is highest in the row center and lowest at the aisles.
        This naturally creates a horizontal 'pressure' toward the aisles.
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
        
        # Pre-calculate distance to nearest aisle center
        aisle_xs = np.array([a["x"] for a in self.cfg.aisles])
        # x_dist_to_aisles: (N_active, N_aisles)
        x_dist_to_aisles = np.abs(x[:, np.newaxis] - aisle_xs[np.newaxis, :])
        min_aisle_dist = np.min(x_dist_to_aisles, axis=1)
        best_aisle_idx = np.argmin(x_dist_to_aisles, axis=1)
        
        # Modulation Factor M(x): quadratic ramp from the aisle center (0 at aisle, 1 far away)
        # Decay over 3.0 meters for a smooth horizontal gradient
        decay_dist = 3.0
        m_x = np.clip(min_aisle_dist / decay_dist, 0.0, 1.0)
        potential_envelope = m_x**4
        
        # Modulation Derivative M'(x) for later pressure calculation
        # Fx = - dU/dx = - U_y * 2 * (x - x_aisle) / decay_dist^2 (for small distances)
        # We simplify this to a lateral 'squeeze'
        lateral_squeeze = np.zeros(len(idx))
        near_aisle_mask = min_aisle_dist < decay_dist
        lateral_squeeze[near_aisle_mask] = (min_aisle_dist[near_aisle_mask] / decay_dist**2)

        for row in self.cfg.seat_rows:
            row_y = row["y"]
            dist_to_row = y - row_y
            abs_dist = np.abs(dist_to_row)
            effective_radius = self.cfg.radius + row["w"]/2
            overlap = np.clip(effective_radius - abs_dist, 0.0, None)
            
            # Vertical Force (Modulated by M(x))
            fy += self.cfg.k_seat * overlap * np.sign(dist_to_row) * potential_envelope
            
            # Horizontal Pressure (Gradient toward lower potential)
            # This pushes agents away from the center of the row toward aisles
            # Sign is away from the nearest aisle center
            push_direction = np.sign(x - aisle_xs[best_aisle_idx])
            # The lateral force is proportional to how much you overlap the seat vertically
            fx -= self.cfg.k_seat * overlap * lateral_squeeze * push_direction
        
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
        # Total forces
        total_force = (
            self._goal_force()
            + self._agent_repulsion()
            + self._wall_force()
            + self.seat_force()
            + self.slope_force()
            + self._fluctuation_force()
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
