from simulation import Config, Simulation
from visualization import plot_simulation_results


def main():
    cfg = Config()
    sim = Simulation(cfg)

    initial_positions = sim.pos.copy()
    times, remaining = sim.run()
    plot_simulation_results(
        cfg=cfg,
        initial_positions=initial_positions,
        snapshots=sim.snapshots,
        times=times,
        remaining=remaining,
    )

    if remaining[-1] == 0:
        print(f"Evacuation completed in {times[-1]:.2f} s")
    else:
        print("Simulation stopped before all agents exited.")


if __name__ == "__main__":
    main()
