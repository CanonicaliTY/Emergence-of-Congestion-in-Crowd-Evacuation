from simulation import Config, Simulation
from visualization import plot_simulation_results, animate_simulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.DataFrame({
        "Run 1" : [],
        "Run 2" : [],
        "Run 3" : [],
        "Run 4" : [],
        "Run 5" : [],
        "Run 6" : [],
        "Run 7" : [],
        "Run 8" : [],
        "Run 9" : [],
        "Run 10" : []
    })
    for j in range(3):
        cfg = Config()
        cfg.n_agents = 200 * (j+3)
        for i in range(1):
            vel_iterator = np.linspace(1, 6, 30)
            exit_times = np.array([])
            cfg.seed = 42 + i

            for vel in vel_iterator:
                cfg.desired_speed = vel
                sim = Simulation(cfg)

                initial_positions = sim.pos.copy()
                times, remaining = sim.run()

                #animate_simulation(cfg, sim.snapshots, times)
                '''      
                plot_simulation_results(
                    cfg=cfg,
                    initial_positions=initial_positions,
                    snapshots=sim.snapshots,
                    times=times,
                    remaining=remaining,
                )
                '''
                if remaining[-1] == 0:
                    print(f"Evacuation completed in {times[-1]:.2f} s")
                    exit_times = np.append(exit_times, times[-1])
                else:
                    print("Simulation stopped before all agents exited.")
                    exit_times = np.append(exit_times, np.inf)
            df[f"Run {i+1}"] = exit_times
            '''
            plt.plot(vel_iterator, exit_times)
            plt.xlabel("Desired Speed")
            plt.ylabel("Exit Time")
            plt.title("Exit Time vs Desired Speed")
            plt.show()
            '''
        df.to_csv(f"exit_times_{(j+3)*200}_agents.csv", index=False)


if __name__ == "__main__":
    main()
