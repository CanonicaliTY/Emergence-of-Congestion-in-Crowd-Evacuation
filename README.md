# Crowd Evacuation Simulation – Simon Theatre E

This project models crowd evacuation from a simplified lecture-theatre geometry
inspired by Simon Theatre E. The core scientific aim is to study how congestion
emerges near a bottleneck during evacuation using a readable 2D agent-based
social-force-style model.

## Recommended Main-Branch Scope

The `main` branch now uses this baseline:

- a single-exit bottleneck geometry
- a 2D model without slope or height-field effects
- one clear default repulsion model
- a simple single-room layout
- static plots as the default output

This keeps the scientific story focused on bottleneck congestion rather than on
multiple competing mechanisms.

## Is the Corridor Necessary?

Not strictly. A bottleneck transition from relatively free flow to congested
flow can already be studied in a room with a single narrow exit.

The corridor is not kept in `main`, because:

- the main scientific question is bottleneck congestion, not the specific
  Simon Theatre E layout
- a single-room, single-exit setup is easier to explain on a poster
- removing the corridor gives a cleaner baseline model with fewer geometric
  details to justify

So the baseline used in `main` is:

- remove the corridor from `main`
- keep the geometry single-room and single-exit
- avoid adding slope effects to the baseline model

## Branch Direction

Currently kept in `main`:

- single-exit bottleneck geometry
- single-room geometry without corridor
- 2D dynamics only
- one default repulsion law
- simple evacuation curve and snapshot plotting

Better suited to `Tingyu` or to a future optional switch:

- multiple exits
- slope or gravity-like spatial field
- alternative repulsion laws for comparison studies
- automatic GIF animation export

The reason for this split is that `main` should stay as the clean reference
model for the congestion study, while `Tingyu` can be the branch for extensions
and sensitivity checks.

## Recent Changes

This section records the main changes and the current decision on each one.

- `multiple exits` had been merged into the code. This has now been removed from
  the default `main` model so that the baseline remains a single bottleneck.
- `corridor-based geometry` is no longer part of the `main` baseline. The model
  now uses a single room with one bottom exit.
- `slope field` had been merged as a gravity-like extra force. This has been
  removed from `main` so the baseline stays purely 2D and easier to interpret.
- `repulsion.py` is kept as a separate module, but `main` now uses one simple
  soft-repulsion law as the default interaction model.
- `animation output` is no longer part of the default run path in `main`.
- `default parameters` have been brought back in line with the simpler baseline
  model rather than the expanded multi-feature version.

## Current File Structure

- `main.py`: current single entry point
- `simulation.py`: simulation engine for the single-room baseline
- `repulsion.py`: baseline repulsion-force implementation
- `visualization.py`: static plotting utilities
- `requirements.txt`: Python dependencies

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The current `main` code runs the baseline simulation and shows the static plots.
