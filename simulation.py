from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # Geometry
    room_w: float = 20.0
    room_h: float = 15.0
    door_w: float = 2.0
    corridor_len: float = 8.0   # used as simplified staircase bottleneck

    # Agents
    n_agents: int = 220
    radius: float = 0.22
    desired_speed: float = 1.6
    corridor_speed_factor: float = 0.8
    tau: float = 0.5

    # Forces
    k_rep: float = 20.0
    k_wall: float = 50.0

    # Time integration
    dt: float = 0.05
    t_max: float = 120.0
    speed_cap: float = 3.0

    # Output
    snapshot_every: int = 40
    seed: int = 42


class Simulation:
    """2D social-force-style evacuation simulation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.room_top = cfg.corridor_len + cfg.room_h
        self.door_xmin = 0.5 * (cfg.room_w - cfg.door_w)
        self.door_xmax = 0.5 * (cfg.room_w + cfg.door_w)

        # Exit point at the bottom of the corridor
        self.exit_point = np.array([cfg.room_w / 2.0, -0.5])

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
        Agents inside the room aim for the door centre.
        Agents inside the corridor aim for the final exit.
        """
        targets = np.tile(self.exit_point, (self.cfg.n_agents, 1))

        in_room = self.pos[:, 1] >= self.cfg.corridor_len
        door_center = np.array([self.cfg.room_w / 2.0, self.cfg.corridor_len - 0.1])
        targets[in_room] = door_center

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
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) < 2:
            return forces

        pos = self.pos[idx]
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

    def _wall_force(self) -> np.ndarray:
        """
        Walls:
        - room left/right/top walls
        - room bottom wall except the door opening
        - corridor side walls
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

        # Outer room boundaries
        fx += self.cfg.k_wall * np.clip(self.cfg.radius - x, 0.0, None)
        fx -= self.cfg.k_wall * np.clip(x - (self.cfg.room_w - self.cfg.radius), 0.0, None)
        fy -= self.cfg.k_wall * np.clip(y - (self.room_top - self.cfg.radius), 0.0, None)

        # Bottom wall of the lecture theatre, except where the door is
        near_room_bottom = (
            (y >= self.cfg.corridor_len - self.cfg.radius)
            & (y <= self.cfg.corridor_len + self.cfg.radius)
        )
        outside_door = (x < self.door_xmin) | (x > self.door_xmax)
        fy += (
            self.cfg.k_wall
            * np.clip((self.cfg.corridor_len + self.cfg.radius) - y, 0.0, None)
            * near_room_bottom
            * outside_door
        )

        # Corridor side walls
        in_corridor = y < self.cfg.corridor_len + self.cfg.radius
        fx += (
            self.cfg.k_wall
            * np.clip((self.door_xmin + self.cfg.radius) - x, 0.0, None)
            * in_corridor
        )
        fx -= (
            self.cfg.k_wall
            * np.clip(x - (self.door_xmax - self.cfg.radius), 0.0, None)
            * in_corridor
        )

        forces[idx, 0] = fx
        forces[idx, 1] = fy
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
        exited = self.active & (self.pos[:, 1] < -self.cfg.radius)
        self.active[exited] = False
        self.vel[~self.active] = 0.0

    def step(self) -> None:
        total_force = (
            self._goal_force()
            + self._agent_repulsion()
            + self._wall_force()
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
