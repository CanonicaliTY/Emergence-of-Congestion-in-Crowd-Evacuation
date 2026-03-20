from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from simulation import Config


def draw_geometry(ax: plt.Axes, cfg: Config) -> None:
    """Draw room walls with gaps for all defined exits."""
    room_bottom = cfg.corridor_len
    room_top = cfg.corridor_len + cfg.room_h

    # 1. Left Wall (x=0)
    left_exits = sorted([e for e in cfg.exits if e["side"] == "left"], key=lambda e: e["y"])
    curr_y = room_bottom
    for ex in left_exits:
        ax.plot([0, 0], [curr_y, ex["y"] - ex["w"]/2], color="black", linewidth=2)
        curr_y = ex["y"] + ex["w"]/2
    ax.plot([0, 0], [curr_y, room_top], color="black", linewidth=2)

    # 2. Right Wall (x=room_w)
    right_exits = sorted([e for e in cfg.exits if e["side"] == "right"], key=lambda e: e["y"])
    curr_y = room_bottom
    for ex in right_exits:
        ax.plot([cfg.room_w, cfg.room_w], [curr_y, ex["y"] - ex["w"]/2], color="black", linewidth=2)
        curr_y = ex["y"] + ex["w"]/2
    ax.plot([cfg.room_w, cfg.room_w], [curr_y, room_top], color="black", linewidth=2)

    # 3. Top Wall (y=room_top)
    top_exits = sorted([e for e in cfg.exits if e["side"] == "top"], key=lambda e: e["x"])
    curr_x = 0
    for ex in top_exits:
        ax.plot([curr_x, ex["x"] - ex["w"]/2], [room_top, room_top], color="black", linewidth=2)
        curr_x = ex["x"] + ex["w"]/2
    ax.plot([curr_x, cfg.room_w], [room_top, room_top], color="black", linewidth=2)

    # 4. Bottom Wall (y=room_bottom)
    bottom_exits = sorted([e for e in cfg.exits if e["side"] == "bottom"], key=lambda e: e["x"])
    curr_x = 0
    for ex in bottom_exits:
        ax.plot([curr_x, ex["x"] - ex["w"]/2], [room_bottom, room_bottom], color="black", linewidth=2)
        curr_x = ex["x"] + ex["w"]/2
    ax.plot([curr_x, cfg.room_w], [room_bottom, room_bottom], color="black", linewidth=2)

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

def animate_simulation(cfg: Config, snapshots: Sequence[np.ndarray], times: np.ndarray) -> None:
    """Create an animation of the evacuation process."""
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_geometry(ax, cfg)
    
    scatter = ax.scatter([], [], s=10, alpha=0.9)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def update(frame):
        if frame < len(snapshots):
            scatter.set_offsets(snapshots[frame])
            time_text.set_text(f'Time: {times[frame]:.2f} s')
        return scatter, time_text
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=125, blit=True
    )
    ani.save("evacuation.gif", writer="pillow", fps=30)
    plt.show()

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
