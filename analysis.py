from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from simulation import Config, RunResults, Simulation


@dataclass
class ParameterScanResult:
    """Mean observables from repeated runs at fixed control-parameter values."""

    parameter_name: str
    parameter_values: np.ndarray
    repeats: int
    mean_evacuation_time: np.ndarray
    std_evacuation_time: np.ndarray
    mean_average_flux: np.ndarray
    std_average_flux: np.ndarray
    mean_door_density: np.ndarray
    std_door_density: np.ndarray
    mean_clogging_fraction: np.ndarray
    std_clogging_fraction: np.ndarray
    completion_fraction: np.ndarray
    density_values: np.ndarray | None = None

    @property
    def x_values(self) -> np.ndarray:
        if self.parameter_name == "n_agents" and self.density_values is not None:
            return self.density_values
        return self.parameter_values

    @property
    def x_label(self) -> str:
        if self.parameter_name == "n_agents":
            return "Global density N / A_room"
        if self.parameter_name == "door_w":
            return "Door width"
        if self.parameter_name == "desired_speed":
            return "Desired speed v0"
        return self.parameter_name


@dataclass
class TransitionEstimate:
    """
    Simple finite-size estimate of the onset of congestion.

    This is interpreted as a transition-like crossover point rather than a
    rigorous critical point.
    """

    axis_label: str
    estimated_axis_value: float
    estimated_parameter_value: float
    flux_peak_axis_value: float | None
    flux_peak_parameter_value: float | None
    clogging_onset_axis_value: float | None
    clogging_onset_parameter_value: float | None
    evacuation_growth_axis_value: float | None
    evacuation_growth_parameter_value: float | None
    explanation: str


def summarize_run(results: RunResults) -> dict[str, float | bool]:
    """Convert one run into a compact dictionary of poster-ready observables."""
    return {
        "completed": results.completed,
        "simulation_duration": results.simulation_duration,
        "evacuation_time": results.evacuation_time,
        "average_flux": results.average_flux,
        "mean_door_density": results.mean_door_density,
        "clogging_fraction": results.clogging_fraction,
        "stalled_fraction": results.stalled_fraction,
        "initial_agent_count": results.initial_agent_count,
        "global_density": results.global_density,
    }


def run_single_simulation(cfg: Config) -> RunResults:
    """Run one simulation and return the full time-resolved result object."""
    return Simulation(cfg).run()


def run_single_summary(cfg: Config) -> dict[str, float | bool]:
    """Run one simulation and return only the summary observables."""
    return summarize_run(run_single_simulation(cfg))


def run_repeated_simulations(base_cfg: Config, repeats: int = 5) -> list[dict[str, float | bool]]:
    """Repeat the same configuration with different seeds."""
    summaries = []
    for repeat_index in range(repeats):
        cfg = replace(base_cfg, seed=base_cfg.seed + repeat_index)
        summaries.append(run_single_summary(cfg))
    return summaries


def _nanmean_and_std(values: np.ndarray) -> tuple[float, float]:
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values))


def run_parameter_scan(
    base_cfg: Config,
    parameter_name: str,
    values: Sequence[float],
    repeats: int = 5,
) -> ParameterScanResult:
    """
    Repeat a single-parameter scan with different seeds at each scan point.

    Only one control parameter is changed at a time so the interpretation stays
    simple and suitable for an undergraduate poster.
    """
    parameter_values = np.asarray(values, dtype=float)

    mean_evacuation_time = []
    std_evacuation_time = []
    mean_average_flux = []
    std_average_flux = []
    mean_door_density = []
    std_door_density = []
    mean_clogging_fraction = []
    std_clogging_fraction = []
    completion_fraction = []

    for value_index, value in enumerate(parameter_values):
        summaries = []
        parameter_value = int(round(value)) if parameter_name == "n_agents" else float(value)
        for repeat_index in range(repeats):
            seed = base_cfg.seed + value_index * repeats + repeat_index
            cfg = replace(base_cfg, seed=seed, **{parameter_name: parameter_value})
            summaries.append(run_single_summary(cfg))

        evacuation_times = np.array([summary["evacuation_time"] for summary in summaries], dtype=float)
        average_fluxes = np.array([summary["average_flux"] for summary in summaries], dtype=float)
        door_densities = np.array([summary["mean_door_density"] for summary in summaries], dtype=float)
        clogging_fractions = np.array([summary["clogging_fraction"] for summary in summaries], dtype=float)
        completed = np.array([summary["completed"] for summary in summaries], dtype=float)

        mean_value, std_value = _nanmean_and_std(evacuation_times)
        mean_evacuation_time.append(mean_value)
        std_evacuation_time.append(std_value)

        mean_value, std_value = _nanmean_and_std(average_fluxes)
        mean_average_flux.append(mean_value)
        std_average_flux.append(std_value)

        mean_value, std_value = _nanmean_and_std(door_densities)
        mean_door_density.append(mean_value)
        std_door_density.append(std_value)

        mean_value, std_value = _nanmean_and_std(clogging_fractions)
        mean_clogging_fraction.append(mean_value)
        std_clogging_fraction.append(std_value)

        completion_fraction.append(float(np.mean(completed)))

    density_values = None
    if parameter_name == "n_agents":
        density_values = parameter_values / (base_cfg.room_w * base_cfg.room_h)

    return ParameterScanResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        repeats=repeats,
        mean_evacuation_time=np.asarray(mean_evacuation_time, dtype=float),
        std_evacuation_time=np.asarray(std_evacuation_time, dtype=float),
        mean_average_flux=np.asarray(mean_average_flux, dtype=float),
        std_average_flux=np.asarray(std_average_flux, dtype=float),
        mean_door_density=np.asarray(mean_door_density, dtype=float),
        std_door_density=np.asarray(std_door_density, dtype=float),
        mean_clogging_fraction=np.asarray(mean_clogging_fraction, dtype=float),
        std_clogging_fraction=np.asarray(std_clogging_fraction, dtype=float),
        completion_fraction=np.asarray(completion_fraction, dtype=float),
        density_values=density_values,
    )


def scan_agent_counts(
    base_cfg: Config,
    agent_counts: Sequence[int],
    repeats: int = 5,
) -> ParameterScanResult:
    return run_parameter_scan(base_cfg, "n_agents", agent_counts, repeats=repeats)


def scan_door_widths(
    base_cfg: Config,
    door_widths: Sequence[float],
    repeats: int = 5,
) -> ParameterScanResult:
    return run_parameter_scan(base_cfg, "door_w", door_widths, repeats=repeats)


def scan_desired_speeds(
    base_cfg: Config,
    desired_speeds: Sequence[float],
    repeats: int = 5,
) -> ParameterScanResult:
    return run_parameter_scan(base_cfg, "desired_speed", desired_speeds, repeats=repeats)


def estimate_congestion_onset(
    scan: ParameterScanResult,
    clogging_threshold: float = 0.05,
    relative_evacuation_growth: float = 0.15,
) -> TransitionEstimate:
    """
    Estimate a transition-like onset from a crowd-size or density scan.

    The default interpretation is modest:
    - first look for clearly non-zero clogging time
    - otherwise use the first sharp rise in evacuation time
    - always report the flux maximum as a supporting reference point
    """
    axis_values = scan.x_values

    flux_peak_index = None
    if not np.all(np.isnan(scan.mean_average_flux)):
        flux_peak_index = int(np.nanargmax(scan.mean_average_flux))

    clogging_candidates = np.where(scan.mean_clogging_fraction >= clogging_threshold)[0]
    clogging_index = int(clogging_candidates[0]) if len(clogging_candidates) > 0 else None

    growth_index = None
    previous_times = scan.mean_evacuation_time[:-1]
    current_times = scan.mean_evacuation_time[1:]
    valid_growth = (~np.isnan(previous_times)) & (~np.isnan(current_times)) & (previous_times > 0.0)
    if np.any(valid_growth):
        relative_growth = np.full(len(current_times), np.nan)
        relative_growth[valid_growth] = (
            (current_times[valid_growth] - previous_times[valid_growth])
            / previous_times[valid_growth]
        )
        growth_candidates = np.where(relative_growth >= relative_evacuation_growth)[0]
        if len(growth_candidates) > 0:
            growth_index = int(growth_candidates[0] + 1)

    if clogging_index is not None:
        estimate_index = clogging_index
        explanation = (
            "Estimated from the first scan point where the mean clogging fraction "
            f"exceeds {clogging_threshold:.2f}. This is interpreted as the onset "
            "of sustained congestion near the doorway."
        )
    elif growth_index is not None:
        estimate_index = growth_index
        explanation = (
            "Estimated from the first sharp rise in mean evacuation time "
            f"(relative increase >= {relative_evacuation_growth:.2f}). This is "
            "interpreted as a transition-like crossover into stronger congestion."
        )
    elif flux_peak_index is not None:
        estimate_index = flux_peak_index
        explanation = (
            "Estimated from the scan point where the mean average flux reaches "
            "its maximum. This is treated as a practical marker for the onset "
            "of reduced throughput."
        )
    else:
        estimate_index = 0
        explanation = "No clear onset was found, so the first scan point is returned as a placeholder."

    return TransitionEstimate(
        axis_label=scan.x_label,
        estimated_axis_value=float(axis_values[estimate_index]),
        estimated_parameter_value=float(scan.parameter_values[estimate_index]),
        flux_peak_axis_value=(
            float(axis_values[flux_peak_index])
            if flux_peak_index is not None
            else None
        ),
        flux_peak_parameter_value=(
            float(scan.parameter_values[flux_peak_index])
            if flux_peak_index is not None
            else None
        ),
        clogging_onset_axis_value=(
            float(axis_values[clogging_index])
            if clogging_index is not None
            else None
        ),
        clogging_onset_parameter_value=(
            float(scan.parameter_values[clogging_index])
            if clogging_index is not None
            else None
        ),
        evacuation_growth_axis_value=(
            float(axis_values[growth_index])
            if growth_index is not None
            else None
        ),
        evacuation_growth_parameter_value=(
            float(scan.parameter_values[growth_index])
            if growth_index is not None
            else None
        ),
        explanation=explanation,
    )
