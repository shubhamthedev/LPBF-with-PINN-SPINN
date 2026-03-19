# Studienarbeit — PINN & S-PINN for LPBF Temperature Modelling

> **Work in Progress** — This project is part of an ongoing Studienarbeit and is actively being developed.

---

## What This Is

This project explores the use of **Physics-Informed Neural Networks (PINNs)** and **Separable PINNs (S-PINNs)** to model the temperature field and melt pool dynamics during **Laser Powder Bed Fusion (LPBF)** — a metal additive manufacturing process.

The core idea is to solve the 3D transient heat equation without any simulation data, by embedding the governing physics directly into the neural network's loss function. The material used is **Hastelloy X**, with fully temperature-dependent thermal properties.

---

## Motivation

Traditional numerical solvers (FEM, FDM) for LPBF are accurate but expensive. PINNs offer a mesh-free alternative that can, in principle, be faster to query once trained. S-PINNs go a step further by tackling the **curse of dimensionality** that standard PINNs run into when the number of collocation points grows in 3D+time problems.

This project benchmarks both approaches on the same physical problem under fair, controlled conditions.

---

## Problem Setup

- **Domain** — 3.0 × 1.5 × 1.0 mm (x × y × z)
- **Laser path** — 2.0 mm single track (x = 0.5 mm → 2.5 mm)
- **Scan speed** — 0.1 m/s, absorbed power 150 W
- **Beam radius** — 450 µm, penetration depth 500 µm (Goldak model)
- **Material** — Hastelloy X (temperature-dependent κ and c_p)
- **Initial condition** — uniform T₀ = 300 K
- **Boundary conditions** — Robin (convection) on top surface, Neumann (insulated) on all other faces

---

## Project Structure

```
├── pinn/
│   └── pinn_lpbf.py          # Standard PINN implementation
├── spinn/
│   └── spinn_lpbf.py         # Separable PINN implementation
├── results/
│   ├── snapshots/            # Temperature field plots
│   ├── centreline/           # Centreline temperature profiles
│   └── loss_history/         # Training loss curves
├── reference/                # Ground truth FD solution (from supervisor)
└── README.md
```

---

## Current Status

| Task | Status |
|---|---|
| PINN implementation (single track) | ✅ Done |
| S-PINN implementation | 🔄 In progress |
| Domain updated to 2 mm laser path | ✅ Done |
| Fair comparison setup (hard IC, matched hyperparameters) | ✅ Done |
| Ground truth FD reference solution | ⏳ Awaiting from supervisor |
| Melt pool steady-state analysis | 🔄 In progress |
| Benchmarking PINN vs S-PINN | ⏳ Pending |
| Final report write-up | ⏳ Pending |

---

## How to Run

**Install dependencies**

```bash
pip install torch numpy scipy matplotlib
```

**Train the PINN**

```bash
python pinn/pinn_lpbf.py
```

**Train the S-PINN**

```bash
python spinn/spinn_lpbf.py
```

Results are saved automatically to the `results/` folder.

---

## Key References

- Safari & Wessels (2025) — *PINN and DeepONet for LPBF temperature modelling* (main reference, parameters taken from Table 1)
- Cho et al. (2023) — *Separable Physics-Informed Neural Networks*, NeurIPS 2023
- Goldak et al. (1984) — *Double ellipsoid heat source model*

---

## Notes

- Parameters follow **Safari & Wessels (2025), Table 1** exactly
- The PINN uses a **hard initial condition ansatz** so that T = T₀ at t = 0 is satisfied exactly by construction, not through a loss term
- Melt pool dimensions are measured at the **end of the laser track** (t = 20 ms), where the melt pool is expected to have reached steady state
- Validation against the finite-difference reference solution from the supervisor is still pending

---

*Studienarbeit — Institute for Applied Mechanics, 2026*