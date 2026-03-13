# Crowd Evacuation Simulation – Simon Theatre E

This project investigates how congestion emerges when a large number of people attempt to leave a lecture theatre through a narrow exit. The system is inspired by **Simon Theatre E**, located in the basement of the Simon Building, where lectures often end with noticeable crowding near the exit and staircase.

The goal of the project is to model this situation using a **numerical simulation of interacting agents** and to explore how collective behaviour arises from simple local rules.

---

## Motivation

Crowd motion is an example of **emergent collective behaviour**. Even when individuals simply attempt to move toward an exit, interactions between people can lead to:

- congestion near bottlenecks  
- temporary clogging events  
- nonlinear evacuation dynamics  

These phenomena are closely related to topics studied in physics such as:

- active matter  
- non-equilibrium systems  
- many-body dynamics

---

## Project Context

This repository was created as part of a University of Manchester Physics poster project.

The aim is to connect a familiar real-world situation with concepts from **statistical physics and active matter**.

---

## Model

Each person is represented as an **agent moving in continuous two-dimensional space**. The motion of each agent follows a simplified **social force model**, where individuals attempt to move toward the exit while avoiding collisions with other agents and walls.

The equation of motion for agent \(i\) is

\[
m \frac{d\vec{v}_i}{dt}
=
\vec{F}_i^{goal}
+
\sum_j \vec{F}_{ij}^{repulsion}
+
\vec{F}_i^{wall}
\]

where

- \(F^{goal}\) drives motion toward the exit
- \(F^{repulsion}\) prevents agents from overlapping
- \(F^{wall}\) prevents agents from crossing boundaries

---

## Objectives

The simulation allows us to explore several questions:

- How evacuation time depends on **crowd density**
- How congestion forms near **narrow exits**
- Whether spontaneous **clogging** occurs
- Whether increasing walking speed can lead to the **faster-is-slower effect**

---

## Simulation Setup

The simulated environment represents a simplified version of the layout of Simon Theatre E:

- a lecture theatre region where agents initially sit
- a narrow exit doorway
- a corridor/staircase region leading out of the building

Typical simulations use:

- **200–300 agents**
- a **time-stepping numerical integration**
- simple geometric boundaries

The simulation runs until all agents have exited the system.

---

## Visualisations

The simulation generates several outputs useful for analysing crowd behaviour:

- agent trajectories
- density heatmaps
- evacuation curves (people remaining vs time)
- parameter scans for different crowd sizes or exit widths

---

## Technologies

The project is implemented in **Python** using:

- `numpy` for numerical computation  
- `matplotlib` for visualisation  

---

## How to Use

---

## Updates

---
