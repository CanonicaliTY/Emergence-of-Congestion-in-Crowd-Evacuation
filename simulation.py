from dataclasses import dataclass

import numpy as np

from repulsion import Repulsion

from scipy.spatial import cKDTree

@dataclass(frozen=True)
class ObservationRegion:
    """Rectangular area used to measure crowding directly above the door."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclass
class Config:
    # Geometry
    room_w: float = 20.0
    room_h: float = 15.0
    door_w: float = 2.0

    # Agents
    n_agents: int = 250 
    radius: float = 0.22
    desired_speed: float = 1.2
    tau: float = 0.5

    # Forces
    k_simple_rep: float = 20.0
    k_wall: float = 50.0

    # Time integration
    dt: float = 0.005
    t_max: float = 250.0
    speed_cap: float = 2.0

    # Analysis
    door_region_width_margin: float = 0.75
    door_region_height: float = 1.5
    outflow_bin_width: float = 1.0
    clogging_density_threshold: float = 3.0
    low_outflow_threshold: float = 0.2

    # Output
    snapshot_every: int = 40
    seed: int = 42


@dataclass
class RunResults:
    """Time series and summary observables for one evacuation run."""

    config: Config
    initial_positions: np.ndarray
    snapshots: list[np.ndarray]
    times: np.ndarray
    remaining: np.ndarray
    cumulative_evacuated: np.ndarray
    step_outflow: np.ndarray
    door_density: np.ndarray
    outflow_times: np.ndarray
    binned_outflow: np.ndarray
    binned_door_density: np.ndarray
    clogged_bins: np.ndarray
    stalled_bins: np.ndarray
    door_region: ObservationRegion
    simulation_duration: float
    evacuation_time: float
    average_flux: float
    mean_door_density: float
    clogging_fraction: float
    stalled_fraction: float
    completed: bool
    initial_agent_count: int
    global_density: float
    peak_congestion: tuple


class Simulation:
    """Single-room, single-exit 2D evacuation simulation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.room_top = cfg.room_h
        self.door_xmin = 0.5 * (cfg.room_w - cfg.door_w)
        self.door_xmax = 0.5 * (cfg.room_w + cfg.door_w)
        self.exit_point = np.array([cfg.room_w / 2.0, -0.5])
        self.door_region = self._build_door_region()
        self.repulsion = Repulsion(cfg)

        self.pos = self._initialize_agents()
        self.initial_positions = self.pos.copy()
        self.vel = np.zeros_like(self.pos)
        self.active = np.ones(cfg.n_agents, dtype=bool)

        self.times = [0.0]
        self.remaining = [cfg.n_agents]
        self.exit_counts = [0]
        self.door_density_series = [self._measure_door_density()]
        self.snapshots = [self.pos[self.active].copy()]

    def _active_indices(self) -> np.ndarray:
        return np.where(self.active)[0]

    def _build_door_region(self) -> ObservationRegion:
        """Build the rectangular observation region used for local density."""
        xmin = max(0.0, self.door_xmin - self.cfg.door_region_width_margin)
        xmax = min(self.cfg.room_w, self.door_xmax + self.cfg.door_region_width_margin)
        return ObservationRegion(
            xmin=xmin,
            xmax=xmax,
            ymin=0.0,
            ymax=min(self.room_top, self.cfg.door_region_height),
        )

    def _initialize_agents(self) -> np.ndarray:
        """Randomly place agents inside the room while avoiding large overlaps."""
        rng = np.random.default_rng(self.cfg.seed)
        pos = np.zeros((self.cfg.n_agents, 2))

        count = 0
        attempts = 0
        max_attempts = 200000

        while count < self.cfg.n_agents and attempts < max_attempts:
            candidate = np.array(
                [
                    rng.uniform(self.cfg.radius, self.cfg.room_w - self.cfg.radius),
                    rng.uniform(self.cfg.radius, self.room_top - self.cfg.radius),
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
        """All active agents move toward the single exit below the door."""
        return np.tile(self.exit_point, (self.cfg.n_agents, 1))

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

        desired_vel = self.cfg.desired_speed * directions
        forces[idx] = (desired_vel - vel) / self.cfg.tau
        return forces

    def _agent_repulsion(self) -> np.ndarray:
        """Simple soft repulsion used as the default crowd-interaction model."""
        return self.repulsion.simple_agent_repulsion(self.pos, self._active_indices())

    def _wall_force(self) -> np.ndarray:
        """Room walls with one opening on the bottom wall."""
        forces = np.zeros_like(self.pos)
        idx = self._active_indices()
        if len(idx) == 0:
            return forces

        pos = self.pos[idx]
        x = pos[:, 0]
        y = pos[:, 1]

        fx = np.zeros(len(idx))
        fy = np.zeros(len(idx))

        fx += self.cfg.k_wall * np.clip(self.cfg.radius - x, 0.0, None)
        fx -= self.cfg.k_wall * np.clip(x - (self.cfg.room_w - self.cfg.radius), 0.0, None)
        fy -= self.cfg.k_wall * np.clip(y - (self.room_top - self.cfg.radius), 0.0, None)

        near_bottom = y <= self.cfg.radius
        outside_door = (x < self.door_xmin) | (x > self.door_xmax)
        fy += self.cfg.k_wall * np.clip(self.cfg.radius - y, 0.0, None) * near_bottom * outside_door

        forces[idx, 0] = fx
        forces[idx, 1] = fy
        return forces

    def _measure_door_density(self) -> float:
        """
        Local density above the door, measured as agents per unit area inside a
        fixed rectangular observation region.
        """
        idx = self._active_indices()
        if len(idx) == 0:
            return 0.0

        pos = self.pos[idx]
        in_region = (
            (pos[:, 0] >= self.door_region.xmin)
            & (pos[:, 0] <= self.door_region.xmax)
            & (pos[:, 1] >= self.door_region.ymin)
            & (pos[:, 1] <= self.door_region.ymax)
        )
        return float(np.count_nonzero(in_region) / self.door_region.area)

    def _limit_speed(self) -> None:
        idx = self._active_indices()
        if len(idx) == 0:
            return

        speeds = np.linalg.norm(self.vel[idx], axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-8)
        factors = np.minimum(1.0, self.cfg.speed_cap / speeds)
        self.vel[idx] *= factors

    def _remove_exited_agents(self) -> int:
        exited = self.active & (self.pos[:, 1] < -self.cfg.radius)
        exited_count = int(np.count_nonzero(exited))
        self.active[exited] = False
        self.vel[~self.active] = 0.0
        return exited_count

    def _build_bin_edges(self, final_time: float) -> np.ndarray:
        if final_time <= 0.0:
            return np.array([0.0, self.cfg.outflow_bin_width], dtype=float)

        edges = np.arange(0.0, final_time + self.cfg.outflow_bin_width, self.cfg.outflow_bin_width)
        if len(edges) < 2:
            edges = np.array([0.0, self.cfg.outflow_bin_width], dtype=float)
        elif edges[-1] < final_time:
            edges = np.append(edges, edges[-1] + self.cfg.outflow_bin_width)
        return edges

    def _compute_binned_observables(
        self,
        times: np.ndarray,
        exit_counts: np.ndarray,
        door_density: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = self._build_bin_edges(float(times[-1]))
        outflow_times = 0.5 * (edges[:-1] + edges[1:])

        step_times = times[1:]
        step_exit_counts = exit_counts[1:]
        exits_per_bin, _ = np.histogram(step_times, bins=edges, weights=step_exit_counts)
        binned_outflow = exits_per_bin / np.diff(edges)

        density_sum, _ = np.histogram(times, bins=edges, weights=door_density)
        density_count, _ = np.histogram(times, bins=edges)
        binned_door_density = np.divide(
            density_sum,
            density_count,
            out=np.zeros(len(edges) - 1, dtype=float),
            where=density_count > 0,
        )

        return outflow_times, binned_outflow, binned_door_density

    def _build_results(self) -> RunResults:
        times = np.array(self.times, dtype=float)
        remaining = np.array(self.remaining, dtype=int)
        exit_counts = np.array(self.exit_counts, dtype=float)
        door_density = np.array(self.door_density_series, dtype=float)

        cumulative_evacuated = self.cfg.n_agents - remaining
        step_outflow = exit_counts / self.cfg.dt
        outflow_times, binned_outflow, binned_door_density = self._compute_binned_observables(
            times=times,
            exit_counts=exit_counts,
            door_density=door_density,
        )

        # The main congestion/clogging indicator is high local door density.
        # A stricter stalled indicator also checks for low outflow in the same bin.
        clogged_bins = binned_door_density >= self.cfg.clogging_density_threshold
        stalled_bins = clogged_bins & (binned_outflow <= self.cfg.low_outflow_threshold)

        completed = bool(remaining[-1] == 0)
        simulation_duration = float(times[-1])
        evacuation_time = simulation_duration if completed else np.nan
        average_flux = (
            self.cfg.n_agents / evacuation_time
            if completed and evacuation_time > 0.0
            else np.nan
        )

        return RunResults(
            config=self.cfg,
            initial_positions=self.initial_positions,
            snapshots=list(self.snapshots),
            times=times,
            remaining=remaining,
            cumulative_evacuated=cumulative_evacuated,
            step_outflow=step_outflow,
            door_density=door_density,
            outflow_times=outflow_times,
            binned_outflow=binned_outflow,
            binned_door_density=binned_door_density,
            clogged_bins=clogged_bins,
            stalled_bins=stalled_bins,
            door_region=self.door_region,
            simulation_duration=simulation_duration,
            evacuation_time=evacuation_time,
            average_flux=average_flux,
            mean_door_density=float(np.mean(door_density)),
            clogging_fraction=float(np.mean(clogged_bins)),
            stalled_fraction=float(np.mean(stalled_bins)),
            completed=completed,
            initial_agent_count=self.cfg.n_agents,
            global_density=self.cfg.n_agents / (self.cfg.room_w * self.cfg.room_h),
            peak_congestion=self.congestion
        )

    def step(self) -> int:
        total_force = (
            self._goal_force()
            + self._agent_repulsion()
            + self._wall_force()
        )

        idx = self._active_indices()
        if len(idx) == 0:
            return 0

        self.vel[idx] += total_force[idx] * self.cfg.dt
        self._limit_speed()
        self.pos[idx] += self.vel[idx] * self.cfg.dt
        return self._remove_exited_agents()
    
    def congestion_level(self) -> float:
        idx = self._active_indices()    
        ROI = [i for i in idx if  self.door_region.xmin <= self.pos[i][0] <= self.door_region.xmax  and  self.door_region.ymin <= self.pos[i][1] <= self.door_region.ymax]
        positions = self.pos[ROI]
        N, d = positions.shape
        if N < 4:
            return tuple((N,0))
        velocities = self.vel[ROI]
        speeds = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
        denom = max(float(np.mean(speeds)), 1e-8)
        tree = cKDTree(positions)
        distances, idxs = tree.query(positions, k=N)
        if distances.shape[1] > 1:
            nn = distances[:, 1]
        else:
            # compute nearest nonzero distance
            all_dists = np.sqrt(((positions[:, None, :] - positions[None, :, :])**2).sum(-1))
            all_dists[all_dists == 0] = np.inf
            nn = np.min(all_dists, axis=1)
        h = max(np.median(nn), 1e-8)
        curls = np.full((N,), np.nan, dtype=float)
        for i in range(N):
            neigh_idx = idxs[i]
            neigh_idx = np.unique(neigh_idx)  # keep unique neighbors (includes self)
            X = positions[neigh_idx] - positions[i]      # (m,2)
            V = velocities[neigh_idx] - velocities[i]   # (m,2)

            r2 = np.sum(X * X, axis=1)
            w = np.exp(-0.5 * r2 / (h * h))

            if X.shape[0] < 4 or np.sum(w) <= 1e-12:
                continue

            sw = np.sqrt(w)[:, None]
            A = sw * X  # (m,2)
            J = np.zeros((2, 2), dtype=float)
            # solve for each velocity component: A @ p = b
            for comp in range(2):
                b = sw[:, 0] * V[:, comp]
                p, *_ = np.linalg.lstsq(A, b, rcond=None)
                J[comp, :] = p

            # scalar curl z = d v_y / dx - d v_x / dy = J[1,0] - J[0,1]
            curls[i] = J[1, 0] - J[0, 1]
        return tuple((N, ((np.max(curls)-np.min(curls))/denom)))


    def run(self) -> RunResults:
        n_steps = int(self.cfg.t_max / self.cfg.dt)
        ct = 0
        self.congestion = tuple((0,0))

        for step in range(1, n_steps + 1):
            if not np.any(self.active):
                break

            exited_count = self.step()

            cg = self.congestion_level()
            if(np.abs(cg[1]) > np.abs(ct)):
                ct = cg[1]
                self.congestion = cg 

            current_time = step * self.cfg.dt
            n_remaining = int(np.sum(self.active))

            self.times.append(current_time)
            self.remaining.append(n_remaining)
            self.exit_counts.append(exited_count)
            self.door_density_series.append(self._measure_door_density())

            if step % self.cfg.snapshot_every == 0:
                self.snapshots.append(self.pos[self.active].copy())

        self.snapshots.append(self.pos[self.active].copy())
        return self._build_results()
