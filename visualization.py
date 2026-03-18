from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from simulation import Config


def draw_geometry(ax: plt.Axes, cfg: Config) -> None:
    """Draw room, door, corridor walls, and exit line."""
    room_bottom = cfg.corridor_len
    room_top = cfg.corridor_len + cfg.room_h
    door_xmin = 0.5 * (cfg.room_w - cfg.door_w)
    door_xmax = 0.5 * (cfg.room_w + cfg.door_w)

    # Room walls
    ax.plot([0, 0], [room_bottom, room_top], linewidth=2)
    ax.plot([cfg.room_w, cfg.room_w], [room_bottom, room_top], linewidth=2)
    ax.plot([0, cfg.room_w], [room_top, room_top], linewidth=2)

    # Bottom room wall with a gap for the door
    ax.plot([0, door_xmin], [room_bottom, room_bottom], linewidth=2)
    ax.plot([door_xmax, cfg.room_w], [room_bottom, room_bottom], linewidth=2)

    # Corridor walls
    ax.plot([door_xmin, door_xmin], [0, room_bottom], linewidth=2)
    ax.plot([door_xmax, door_xmax], [0, room_bottom], linewidth=2)

    # Exit line
    ax.plot([door_xmin, door_xmax], [0, 0], linestyle="--", linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(-1, cfg.room_w + 1)
    ax.set_ylim(-1, room_top + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _last_nonempty_snapshot(
    initial_positions: np.ndarray, snapshots: Sequence[np.ndarray]
) -> np.ndarray:
    for snap in reversed(snapshots):
        if len(snap) > 0:
            return snap
    return initial_positions


def plot_simulation_results(
    cfg: Config,
    initial_positions: np.ndarray,
    snapshots: Sequence[np.ndarray],
    times: np.ndarray,
    remaining: np.ndarray,
) -> None:
    """Plot crowd positions and evacuation curve."""
    last_nonempty_snapshot = _last_nonempty_snapshot(initial_positions, snapshots)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    draw_geometry(ax, cfg)
    ax.scatter(
        initial_positions[:, 0],
        initial_positions[:, 1],
        s=10,
        alpha=0.25,
        label="initial",
    )
    ax.scatter(
        last_nonempty_snapshot[:, 0],
        last_nonempty_snapshot[:, 1],
        s=10,
        alpha=0.9,
        label="late snapshot",
    )
    ax.set_title("Crowd positions")
    ax.legend()

    ax = axes[1]
    ax.plot(times, remaining)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("people remaining")
    ax.set_title("Evacuation curve")

    fig.tight_layout()
    plt.show()
