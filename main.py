import argparse

import matplotlib.pyplot as plt

from analysis import (
    ParameterScanResult,
    TransitionEstimate,
    estimate_congestion_onset,
    scan_agent_counts,
    scan_desired_speeds,
    scan_door_widths,
)
from simulation import Config, RunResults, Simulation
from visualization import (
    plot_density_scan_summary,
    plot_scan_evacuation_time,
    plot_single_run_results,
)

DEFAULT_AGENT_COUNTS = [80, 120, 160, 200, 240, 280]
DEFAULT_DOOR_WIDTHS = [1.2, 1.6, 2.0, 2.4, 2.8]
DEFAULT_DESIRED_SPEEDS = [1.0, 1.3, 1.6, 1.9, 2.2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crowd evacuation analysis")
    parser.add_argument(
        "--mode",
        choices=("single", "density-scan", "door-scan", "speed-scan", "congestion-scan"),
        default="single",
        help="Select a single run or one of the parameter scans.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Number of repeated runs per scan point.",
    )
    return parser


def print_single_run_summary(results: RunResults) -> None:
    print(f"Completed evacuation: {results.completed}")
    print(f"Simulation duration: {results.simulation_duration:.2f} s")
    if results.completed:
        print(f"Total evacuation time T_evac: {results.evacuation_time:.2f} s")
        print(f"Average flux Q = N / T_evac: {results.average_flux:.3f}")
    else:
        print("Total evacuation time T_evac: not reached before t_max")
        print("Average flux Q = N / T_evac: undefined for incomplete evacuation")
    print(f"Mean door density: {results.mean_door_density:.3f}")
    print(f"Clogging fraction: {results.clogging_fraction:.3f}")
    print(f"Stalled clogging fraction: {results.stalled_fraction:.3f}")


def print_scan_summary(
    scan: ParameterScanResult,
    transition: TransitionEstimate | None = None,
) -> None:
    print(f"Scan parameter: {scan.parameter_name}")
    print(f"Repeats per point: {scan.repeats}")
    for index, parameter_value in enumerate(scan.parameter_values):
        x_value = scan.x_values[index]
        print(
            f"value={parameter_value:.3f}, axis={x_value:.3f}, "
            f"T={scan.mean_evacuation_time[index]:.3f}, "
            f"Q={scan.mean_average_flux[index]:.3f}, "
            f"rho_door={scan.mean_door_density[index]:.3f}, "
            f"clog={scan.mean_clogging_fraction[index]:.3f}, "
            f"completion={scan.completion_fraction[index]:.2f}"
        )

    if transition is not None:
        print(
            "Onset-of-congestion estimate: "
            f"{transition.estimated_axis_value:.4f} ({transition.axis_label})"
        )
        print(transition.explanation)


def run_single_mode(cfg: Config) -> None:
    results = Simulation(cfg).run()
    plot_single_run_results(results)
    print_single_run_summary(results)

def run_congestion_scan_mode(cfg: Config, repeats: int) -> None:
    ns = []
    cgs = []
    for N in range(0, 250, 10):
        cfg.n_agents = N
        results = Simulation(cfg).run()
        ns.append(N)
        cgs.append(results.peak_congestion[1])
        print(f"N={results.peak_congestion[0]}, congestion={results.peak_congestion[1]}")
    plt.plot(ns, cgs)
    plt.show()

def run_density_scan_mode(cfg: Config, repeats: int) -> None:
    scan = scan_agent_counts(cfg, DEFAULT_AGENT_COUNTS, repeats=repeats)
    transition = estimate_congestion_onset(scan)
    plot_density_scan_summary(scan, transition=transition)
    print_scan_summary(scan, transition=transition)


def run_door_scan_mode(cfg: Config, repeats: int) -> None:
    scan = scan_door_widths(cfg, DEFAULT_DOOR_WIDTHS, repeats=repeats)
    plot_scan_evacuation_time(scan)
    print_scan_summary(scan)


def run_speed_scan_mode(cfg: Config, repeats: int) -> None:
    scan = scan_desired_speeds(cfg, DEFAULT_DESIRED_SPEEDS, repeats=repeats)
    plot_scan_evacuation_time(scan)
    print_scan_summary(scan)


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config()

    if args.mode == "single":
        run_single_mode(cfg)
    elif args.mode == "density-scan":
        run_density_scan_mode(cfg, repeats=args.repeats)
    elif args.mode == "door-scan":
        run_door_scan_mode(cfg, repeats=args.repeats)
    elif args.mode == "congestion-scan":
        run_congestion_scan_mode(cfg, repeats=args.repeats)
    else:
        run_speed_scan_mode(cfg, repeats=args.repeats)


if __name__ == "__main__":
    main()
