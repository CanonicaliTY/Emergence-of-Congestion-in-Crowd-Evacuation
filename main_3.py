from simulation import Config, Simulation
from visualization import plot_simulation_results, animate_simulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
  cfg = Config()
  cfg.desired_speed = 2.0
  sim = Simulation(cfg)
  
  initial_positions = sim.pos.copy()
  times, remaining = sim.run()

  animate_simulation(cfg, sim.snapshots, times)
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
          
if __name__ == "__main__":
    main()
