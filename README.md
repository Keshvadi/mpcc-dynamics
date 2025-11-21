# Performance Analysis of Dynamic Equilibria in Joint Path Selection and Congestion Control in Path-Aware Networks

This repository contains the simulation code and analytical models for the paper "Performance Analysis of Dynamic Equilibria in Joint Path Selection and Congestion Control in Path-Aware Networks".

## Overview
The project implements a discrete-event simulator to validate theoretical findings regarding network oscillations caused by greedy path selection combined with loss-based congestion control. It includes:
* Deterministic expected dynamics model (Python)
* Stochastic discrete-event simulator (Python)
* Scripts to reproduce all figures in the paper

## Interactive Demo
An interactive version of the simulation is available in Google Colab, allowing users to tweak parameters and visualize results immediately:
* [Open in Google Colab](https://colab.research.google.com/drive/1Z4L5JDeMt09XqoPKO16TyeVqGPiRqaa9?usp=sharing)

The standard Jupyter Notebook file (`mpcc_dynamics.ipynb`) is also included in this repository for local use.

## Requirementsope
* Python 3.8+
* NumPy
* Matplotlib

## Usage
To run the core simulation and generate the validation plots locally:
```bash
./run_all.sh
```
