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
        {"x": 10.0, "y": 8.0, "w": 1.5, "side" : "bottom"}
        #{"x":20.0, "y":21.0, "w":1.0, "side" : "right"}
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
    n_agents: int = 200
    radius: float = 0.20
    desired_speed: float = 1.5
    corridor_speed_factor: float = 0.5
    tau: float = 0.4

    k_simple_rep: float = 1000.0
    # Forces (Helbing et al. 2000) - Mass-normalized
    A_social: float = 0.5          # Social force amplitude (m/s^2)
    B_social: float = 0.08         # Social force range (m)
    k_body: float = 1200.0         # Body compression stiffness
    kappa: float = 2400.0          # Tangential friction coefficient
    k_wall: float = 2000.0         # Walls harder than agents
    k_wall_friction: float = 2400.0 # Match agent friction
    k_seat: float = 100.0
    k_fluct: float = 2.0           # Moderate jitter to break jams
    gamma_damp: float = 500.0      # Damping coefficient for agent-agent forces

    # Time integration
    dt: float = 0.005
    t_max: float = 500.0
    speed_cap: float = 10.0
    max_stuck_time: float = 20.0  # seconds until stuck agent is killed

    # Output
    snapshot_every: int = 10
    seed: int = 42


class Simulation:
    """2D social-force-style evacuation simulation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.room_top = cfg.corridor_len + cfg.room_h

        # Heterogeneous agent sizes
        rng = np.random.default_rng(self.cfg.seed)
        self.radii = rng.normal(cfg.radius, 0.02, size=cfg.n_agents)
        self.radii = np.clip(self.radii, 0.18, 0.28)

        self.pos = self._initialize_agents()
        self.vel = np.zeros_like(self.pos)
        self.active = np.ones(cfg.n_agents, dtype=bool)
        
        self.stuck_time = np.zeros(cfg.n_agents)
        self.kill_timer = np.zeros(cfg.n_agents)
        self.kill_anchor_pos = self.pos.copy()

        # Target caching: only recompute when agent moves >0.5m from last assignment
        self.cached_targets = np.zeros((cfg.n_agents, 2))
        self.target_update_pos = np.full((cfg.n_agents, 2), np.inf)  # inf forces first-pass compute
        # Per-agent fixed noise offsets — assigned once, stable across timesteps
        rng_noise = np.random.default_rng(cfg.seed + 9999)
        self.agent_exit_noise = rng_noise.normal(0, 0.3, size=cfg.n_agents)
        self.agent_aisle_noise = rng_noise.normal(0, 0.5, size=cfg.n_agents)  # was std=3, now 0.5

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
            r_c = self.radii[count]
            candidate = np.array(
                [
                    rng.uniform(r_c, self.cfg.room_w - r_c),
                    rng.uniform(
                        self.cfg.corridor_len + r_c,
                        self.room_top - r_c,
                    ),
                ]
            )

            if count == 0:
                pos[count] = candidate
                count += 1
            else:
                distances = np.linalg.norm(pos[:count] - candidate, axis=1)
                safe_dists = self.radii[:count] + r_c + 0.01
                if np.all(distances > safe_dists):
                    pos[count] = candidate
                    count += 1

            attempts += 1

        if count < self.cfg.n_agents:
            raise RuntimeError("Could not place all agents without overlap. Try fewer agents.")

        return pos

    def _targets(self) -> np.ndarray:
        """
        Returns a target position for every active agent.

        Targets are cached per-agent and only recomputed when an agent has
        moved more than 0.5 m from the position at its last assignment.
        This prevents the goal force from flipping direction every timestep
        due to noisy distance recalculation, which was the cause of
        oscillation in the seating rows.

        Noise is drawn once per agent at construction time (fixed offsets)
        rather than from a time-varying seed, so the assignment is stable.
        """
        idx = self._active_indices()
        if len(idx) == 0:
            return self.cached_targets

        # Determine which active agents need a target update
        displacement = np.linalg.norm(self.pos[idx] - self.target_update_pos[idx], axis=1)
        needs_update_mask = displacement > 0.5
        update_idx = idx[needs_update_mask]   # global indices of agents to recompute

        if len(update_idx) == 0:
            return self.cached_targets

        pos = self.pos[update_idx]

        # --- 1. Build exit geometry ---
        exit_centers = []
        offset_targets = []
        for ex in self.cfg.exits:
            center = np.array([ex['x'], ex['y']])
            exit_centers.append(center)
            offset = 1.0
            if   ex["side"] == "bottom": t = center + [0, -offset]
            elif ex["side"] == "top":    t = center + [0,  offset]
            elif ex["side"] == "left":   t = center + [-offset, 0]
            elif ex["side"] == "right":  t = center + [ offset, 0]
            else:                        t = center
            offset_targets.append(t)
        exit_centers   = np.array(exit_centers)
        offset_targets = np.array(offset_targets)

        # --- 2. Nearest exit, with stable per-agent noise ---
        diffs = pos[:, np.newaxis, :] - exit_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Use the pre-generated fixed noise so exit choice never flips mid-step
        dists += self.agent_exit_noise[update_idx, np.newaxis]

        best_exit_indices = np.argmin(dists, axis=1)
        self.cached_targets[update_idx] = offset_targets[best_exit_indices]

        # --- 3. Aisle targeting for agents still in the seating zone ---
        x_active = pos[:, 0]
        y_active = pos[:, 1]

        in_block_1   = (y_active >= 10.0 - 0.2) & (y_active <= 21.0 + 0.2)
        in_block_2   = (y_active >= 23.0 - 0.2) & (y_active <= 28.0 + 0.2)
        in_seating_zone = in_block_1 | in_block_2

        in_any_door = (self._in_any_door_gap("bottom", x_active) |
                       self._in_any_door_gap("top",    x_active) |
                       self._in_any_door_gap("left",   y_active) |
                       self._in_any_door_gap("right",  y_active))
        in_seating_zone &= ~in_any_door

        if np.any(in_seating_zone):
            active_in_aisle = self._in_any_aisle_gap(x_active)
            aisle_xs = np.array([a["x"] for a in self.cfg.aisles])

            sz_idx   = np.where(in_seating_zone)[0]   # local indices within update_idx
            sz_x     = x_active[sz_idx][:, np.newaxis]
            dist_to_aisle = np.abs(sz_x - aisle_xs[np.newaxis, :])

            # Stable per-agent aisle noise (std=0.5 m, was std=3 m time-varying)
            dist_to_aisle += self.agent_aisle_noise[update_idx[sz_idx], np.newaxis]

            best_aisle_idx  = np.argmin(dist_to_aisle, axis=1)
            nearest_aisle_x = aisle_xs[best_aisle_idx]
            global_sz_idx   = update_idx[sz_idx]

            # Agents not yet in an aisle: move sideways toward it
            mask_sideways = ~active_in_aisle[sz_idx]
            if np.any(mask_sideways):
                t_idx = global_sz_idx[mask_sideways]
                self.cached_targets[t_idx, 0] = nearest_aisle_x[mask_sideways]
                self.cached_targets[t_idx, 1] = y_active[sz_idx[mask_sideways]]

            # Agents already in an aisle: keep x locked to aisle centre
            mask_vertical = active_in_aisle[sz_idx]
            if np.any(mask_vertical):
                v_idx = global_sz_idx[mask_vertical]
                self.cached_targets[v_idx, 0] = nearest_aisle_x[mask_vertical]

        # --- 4. Commit update positions for recomputed agents ---
        self.target_update_pos[update_idx] = self.pos[update_idx].copy()

        return self.cached_targets

    def _goal_force(self) -> np.ndarray:
        forces = np.zeros_like(self.pos)

        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        self._targets()  # updates self.cached_targets for any agents that need it
        targets = self.cached_targets[idx]
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

    def _agent_repulsion(self, dx=None, dy=None, dist=None) -> np.ndarray:
        return Repulsion(self.cfg).simple_agent_repulsion(self.pos, self._active_indices(), dx, dy, dist, radii=self.radii[self._active_indices()], vel=self.vel[self._active_indices()])

    def _tangential_friction(self, dx=None, dy=None, dist=None) -> np.ndarray:
        """Helbing tangential friction: kappa * overlap * delta_v_tangential."""
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) < 2:
            return forces
        
        pos = self.pos[idx]
        vel = self.vel[idx]
        
        if dx is None or dy is None or dist is None:
            dx = pos[:, None, 0] - pos[None, :, 0]
            dy = pos[:, None, 1] - pos[None, :, 1]
            dist = np.sqrt(dx**2 + dy**2)
            dist = np.maximum(dist, 1e-6)
            np.fill_diagonal(dist, np.inf)

        ux = dx / dist
        uy = dy / dist

        # Overlap amount (zero when not in contact)
        two_r = self.radii[idx, None] + self.radii[idx[None, :]]
        overlap = np.clip(two_r - dist, 0.0, None)

        # Tangential velocity component
        vt = vel[:, None, :] - vel[None, :, :]
        vt_dot_tangent = vt[:, :, 0] * (-uy) + vt[:, :, 1] * ux
        
        # Friction force scaled by overlap (more compression = more friction)
        ft = -self.cfg.kappa * overlap * vt_dot_tangent
        
        # Sum over neighbors
        fx = np.sum(ft * (-uy), axis=1)
        fy = np.sum(ft * ux, axis=1)
        
        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces

    def _wall_friction(self) -> np.ndarray:
        """Friction at exit door edges — agents catch on the doorframe."""
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        pos = self.pos[idx]
        vel = self.vel[idx]
        r = self.cfg.radius

        fx = np.zeros(len(idx))
        fy = np.zeros(len(idx))

        for ex in self.cfg.exits:
            if ex["side"] in ["bottom", "top"]:
                # Door edges are vertical lines at x = door_x ± w/2
                door_y = ex["y"]
                near_door_y = np.abs(pos[:, 1] - door_y) < 1.0  # within 1m of door
                
                for edge_x in [ex["x"] - ex["w"]/2, ex["x"] + ex["w"]/2]:
                    dist_to_edge = np.abs(pos[:, 0] - edge_x)
                    overlap = np.clip(self.radii[idx] - dist_to_edge, 0.0, None)
                    # Friction opposes vertical velocity (tangential to the edge)
                    fy -= self.cfg.k_wall_friction * overlap * vel[:, 1] * near_door_y
            
            elif ex["side"] in ["left", "right"]:
                door_x = ex["x"]
                near_door_x = np.abs(pos[:, 0] - door_x) < 1.0
                
                for edge_y in [ex["y"] - ex["w"]/2, ex["y"] + ex["w"]/2]:
                    dist_to_edge = np.abs(pos[:, 1] - edge_y)
                    overlap = np.clip(self.radii[idx] - dist_to_edge, 0.0, None)
                    fx -= self.cfg.k_wall_friction * overlap * vel[:, 0] * near_door_x

        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces
        
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
        near_left = x < self.radii[idx]
        fx += self.cfg.k_wall * (self.radii[idx] - x) * near_left * (~self._in_any_door_gap("left", y))

        # Right wall (x=room_w)
        near_right = x > (self.cfg.room_w - self.radii[idx])
        fx -= self.cfg.k_wall * (x - (self.cfg.room_w - self.radii[idx])) * near_right * (~self._in_any_door_gap("right", y))

        # Top wall (y=room_top)
        near_top = y > (self.room_top - self.radii[idx])
        fy -= self.cfg.k_wall * (y - (self.room_top - self.radii[idx])) * near_top * (~self._in_any_door_gap("top", y))

        # Bottom wall (y=corridor_len)
        near_bottom = (y < self.cfg.corridor_len + self.radii[idx]) & (y > self.cfg.corridor_len - self.radii[idx])
        fy += self.cfg.k_wall * ((self.cfg.corridor_len + self.radii[idx]) - y) * near_bottom * (~self._in_any_door_gap("bottom", x))

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
        # Larger kick if stuck > 1.5s
        boost = np.where(self.stuck_time[idx] > 1.5, 10.0, 1.0)
        mag = self.cfg.k_fluct * boost
        
        angles = rng.uniform(0, 2*np.pi, size=len(idx))
        forces[idx, 0] = mag * np.cos(angles)
        forces[idx, 1] = mag * np.sin(angles)
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
            effective_radius = self.radii[idx] + row["w"]/2
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
        idx = self._active_indices()
        if len(idx) == 0:
            return

        speeds = np.linalg.norm(self.vel[idx], axis=1)
        self.stuck_time[idx] = np.where(speeds < 0.2, self.stuck_time[idx] + self.cfg.dt, 0.0)

        # Distance-based kill logic: kill agents if they don't move 1m within max_stuck_time
        self.kill_timer[idx] += self.cfg.dt
        displacement = np.linalg.norm(self.pos[idx] - self.kill_anchor_pos[idx], axis=1)
        
        moved_enough = displacement > 1.0
        if np.any(moved_enough):
            moved_idx = idx[moved_enough]
            self.kill_timer[moved_idx] = 0.0
            self.kill_anchor_pos[moved_idx] = self.pos[moved_idx]

        stuck_too_long = self.kill_timer[idx] > self.cfg.max_stuck_time
        if np.any(stuck_too_long):
            dead_idx = idx[stuck_too_long]
            self.active[dead_idx] = False
            self.vel[dead_idx] = 0.0
            self.kill_timer[dead_idx] = 0.0
            print(f"Killed {np.sum(stuck_too_long)} permanently stuck agent(s) (moved <1m in {self.cfg.max_stuck_time}s). Remaining: {int(np.sum(self.active))}")

        # Update active indices after potential kills
        idx = self._active_indices()
        if len(idx) == 0:
            return

        # Pre-calculate distance matrices for all agents (expensive)
        pos = self.pos[idx]
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.maximum(dist, 1e-6)
        np.fill_diagonal(dist, np.inf)

        # Total forces
        total_force = (
            self._goal_force()
            + self._agent_repulsion(dx, dy, dist)
            + self._wall_force()
            + self._wall_friction()
            + self.seat_force()
            + self.slope_force()
            + self._fluctuation_force()
            + self._tangential_friction(dx, dy, dist)
        )

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
