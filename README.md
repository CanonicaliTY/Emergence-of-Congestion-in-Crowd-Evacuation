# Crowd Evacuation Simulation – Simon Theatre E

This project simulates crowd evacuation from a lecture theatre through a narrow
door and corridor using a simplified 2D agent-based social-force-style model.
It focuses on core evacuation dynamics: goal-directed motion, agent repulsion,
wall interactions, and evacuation over time.

## Project Structure

- `main.py`: single entry point that configures and runs the simulation, then
  calls plotting functions.
- `simulation.py`: simulation engine (`Config` and `Simulation` classes).
- `visualization.py`: plotting utilities for geometry, snapshots, and
  evacuation curve.
- `requirements.txt`: Python dependencies.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The script opens a figure with:

- crowd positions (initial and late snapshot)
- evacuation curve (people remaining vs time)

and prints whether full evacuation completed within `t_max`.
