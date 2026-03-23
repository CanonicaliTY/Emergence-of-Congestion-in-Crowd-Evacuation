import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from analysis import ParameterScanResult, TransitionEstimate
from simulation import Config, ObservationRegion, RunResults


def draw_geometry(ax: plt.Axes, cfg: Config) -> None:
    """Draw the single-room geometry with one door on the bottom wall."""
    room_top = cfg.room_h
    door_xmin = 0.5 * (cfg.room_w - cfg.door_w)
    door_xmax = 0.5 * (cfg.room_w + cfg.door_w)

    ax.plot([0, 0], [0, room_top], linewidth=2)
    ax.plot([cfg.room_w, cfg.room_w], [0, room_top], linewidth=2)
    ax.plot([0, cfg.room_w], [room_top, room_top], linewidth=2)

    ax.plot([0, door_xmin], [0, 0], linewidth=2)
    ax.plot([door_xmax, cfg.room_w], [0, 0], linewidth=2)
    ax.plot([door_xmin, door_xmax], [0, 0], linestyle="--", linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(-1, cfg.room_w + 1)
    ax.set_ylim(-1, room_top + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_observation_region(ax: plt.Axes, region: ObservationRegion) -> None:
    """Outline the rectangle used to measure local density above the door."""
    rectangle = Rectangle(
        (region.xmin, region.ymin),
        region.xmax - region.xmin,
        region.ymax - region.ymin,
        fill=False,
        linestyle=":",
        linewidth=2,
        edgecolor="tab:red",
        label="door density region",
    )
    ax.add_patch(rectangle)


def _last_nonempty_snapshot(results: RunResults) -> np.ndarray:
    for snap in reversed(results.snapshots):
        if len(snap) > 0:
            return snap
    return results.initial_positions


def plot_single_run_results(results: RunResults) -> None:
    """Plot geometry and the main time-resolved observables from one run."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    axes = axes.ravel()

    last_nonempty_snapshot = _last_nonempty_snapshot(results)

    ax = axes[0]
    draw_geometry(ax, results.config)
    draw_observation_region(ax, results.door_region)
    ax.scatter(
        results.initial_positions[:, 0],
        results.initial_positions[:, 1],
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
    ax.set_title("Geometry and crowd positions")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(results.times, results.remaining)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("people remaining")
    ax.set_title("People remaining vs time")

    ax = axes[2]
    ax.plot(results.times, results.cumulative_evacuated)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative evacuated")
    ax.set_title("Cumulative evacuated vs time")

    ax = axes[3]
    ax.plot(results.times, results.door_density, label="door density")
    ax.axhline(
        results.config.clogging_density_threshold,
        linestyle="--",
        color="tab:red",
        label="clogging density threshold",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("density")
    ax.set_title("Local door density vs time")
    ax.legend()

    ax = axes[4]
    ax.bar(
        results.outflow_times,
        results.binned_outflow,
        width=0.85 * results.config.outflow_bin_width,
        color="tab:blue",
        alpha=0.8,
    )
    ax.axhline(
        results.config.low_outflow_threshold,
        linestyle="--",
        color="tab:red",
        label="low-outflow threshold",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("outflow")
    ax.set_title("Time-binned outflow")
    ax.legend()

    ax = axes[5]
    ax.step(
        results.outflow_times,
        results.clogged_bins.astype(int),
        where="mid",
        color="tab:orange",
        label="congested",
    )
    ax.step(
        results.outflow_times,
        results.stalled_bins.astype(int),
        where="mid",
        color="tab:red",
        label="stalled",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state")
    ax.set_yticks([0, 1])
    ax.set_title("Congestion indicator")
    ax.legend()

    fig.tight_layout()
    plt.show()


def _plot_scan_metric(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    transition: TransitionEstimate | None = None,
) -> None:
    ax.errorbar(x_values, y_values, yerr=y_errors, marker="o", capsize=4)
    if transition is not None:
        ax.axvline(
            transition.estimated_axis_value,
            linestyle="--",
            color="tab:red",
            linewidth=1.5,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_density_scan_summary(
    scan: ParameterScanResult,
    transition: TransitionEstimate | None = None,
) -> None:
    """Plot the main averaged observables used to identify congestion onset."""
    x_values = scan.x_values
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    _plot_scan_metric(
        axes[0],
        x_values,
        scan.mean_evacuation_time,
        scan.std_evacuation_time,
        scan.x_label,
        "mean evacuation time (s)",
        "Evacuation time",
        transition=transition,
    )
    _plot_scan_metric(
        axes[1],
        x_values,
        scan.mean_average_flux,
        scan.std_average_flux,
        scan.x_label,
        "mean average flux",
        "Average flux",
        transition=transition,
    )
    _plot_scan_metric(
        axes[2],
        x_values,
        scan.mean_door_density,
        scan.std_door_density,
        scan.x_label,
        "mean door density",
        "Local door density",
        transition=transition,
    )
    _plot_scan_metric(
        axes[3],
        x_values,
        scan.mean_clogging_fraction,
        scan.std_clogging_fraction,
        scan.x_label,
        "mean clogging fraction",
        "Clogging fraction",
        transition=transition,
    )

    fig.tight_layout()
    plt.show()


def plot_scan_evacuation_time(scan: ParameterScanResult) -> None:
    """Plot mean evacuation time against any scanned control parameter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_scan_metric(
        ax,
        scan.x_values,
        scan.mean_evacuation_time,
        scan.std_evacuation_time,
        scan.x_label,
        "mean evacuation time (s)",
        "Evacuation time",
    )
    fig.tight_layout()
    plt.show()
