# Pseudo-spectral DNS of a Collapsing-Core Surrogate for Navier–Stokes

A high-precision pseudo-spectral laboratory investigating the precursor kinematics, scaling laws, and numerical resolution limits of an analytical collapsing-core surrogate inspired by OpenAI's 2026 Navier–Stokes construction.

> **Scope.** This repository studies a **finite-scale, band-limited surrogate** inspired by the paper's construction. It is **not the paper's witness**, **not an independent discretization** of the paper's theorem, and **not a blow-up proof**: the runs stop at a positive clock floor (`q_floor = 0.10`) where the flow is still smooth. The experiments validate the discrete pipeline - solver, forcing, scaling laws - not singularity formation.

---

## 1. Physics & Mathematical Formulation

The simulation solves the 3D incompressible Navier–Stokes equations on the periodic unit torus $\mathbb{T}^3 = [-1/2, 1/2)^3$ with viscosity $\nu$:

$$\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \Delta \mathbf{u} + \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0, \quad \mathbf{u}(\mathbf{x}, 0) = \mathbf{0}$$

driven from rest by the discrete residual of a prescribed reference field $\mathbf{u}^*(\mathbf{x}, t)$:

$$\mathbf{f} = \mathcal{P}\left[\partial_t \mathbf{u}^* + (\mathbf{u}^* \cdot \nabla)\mathbf{u}^* - \nu \Delta \mathbf{u}^*\right]$$

where $\mathcal{P} = \mathbf{I} - \nabla \Delta^{-1} \nabla \cdot$ is the Leray projection onto divergence-free fields.

### The Decoupled / Manufactured Solution Paradigm
Because $\mathbf{u}(\mathbf{x}, 0) = \mathbf{0}$ and $\mathbf{u}^{\ast}(\mathbf{x}, 0) = \mathbf{0}$, the field $\mathbf{u}^{\ast}$ itself is the exact solution of the semi-discrete forced system, and it is the solution the DNS is required to reproduce. This setup is fundamentally a **Method of Manufactured Solutions (MMS)**:
- The solver state receives no feedback from $\mathbf{u}^*$.
- In **analytic mode**, the force generator evaluates $\mathbf{f}(\mathbf{x}, t)$ online.
- In **tabulated mode**, the solver reads precomputed, checksum-verified force snapshots from disk and interpolates in time, having no access to the analytical construction.
- The solver and force generator share identical discrete spectral operators, allowing exact testing of the time integrator under a prescribed collapsing trajectory.

---

## 2. Similarity Coordinates & Flow Kinematics

The reference field is parameterized by similarity time $\tau \equiv 1 - t$. The local collapse clock $q(\mathbf{x}, t)$ is non-uniform in $z$, defined as the positive root of:

$$q = \tau + \left(\frac{z}{L}\right)^2 q^{2h}, \quad 0 < h < \frac{1}{100} \quad (\text{default } h = 0.005)$$

The spatial coordinates scale into dimensionless similarity variables:

$$X = \frac{(r/L)^2}{2q}, \quad \eta = \frac{z}{L q^{1/2 - h}}$$

$$\text{Velocity scaling exponent: } A = \frac{1}{2} + h, \quad \text{Axial scale exponent: } D = \frac{1}{2} - h$$

### Divergence-Free Vector Potential
To ensure $\nabla \cdot \mathbf{u}^* = 0$ exactly in physical space before any projection, the velocity is generated from a vector potential $\mathbf{A} = (-yP, \; xP, \; c) W(r, z)$, where $W(r, z)$ is a smooth $C^\infty$ cutoff window with compact support ($r \le 0.42, |z| \le 0.42$). In cylindrical coordinates $(r, \theta, z)$, $\mathbf{u}^* = \nabla \times \mathbf{A}$ yields:

$$u_r^* = -r \partial_z P, \quad u_\theta^* = -\partial_r c, \quad u_z^* = 2P + r \partial_r P$$

- **Swirl ($u_\theta^*$):** Driven by the toroidal potential $c$, scaling as $u_\theta^* \sim q^{-(1/2 + h)}$.
- **Axial Outflow ($u_z^*$):** Driven by the poloidal potential $P$ with upward bias $j_0$, scaling as $u_z^* \sim q^{-(1/2 + h)} (4\eta + j_0)$.
- **Radial Inflow ($u_r^*$):** Scales as $u_r^* \sim q^{-1/2}$.

```
                        ▲ z (axial outflow u_z ~ q^{-(1/2+h)})
                        │
                  ┌─────┼─────┐
                  │     │     │
                  │   (●│●)   │  <-- Anisotropic Core:
                  │     │     │        r_core ~ q^{1/2}
  ───────────────(──────┼──────)────────► r
                  │     │     │        z_core ~ q^{1/2-h}
                  │   (●│●)   │
                  │     │     │  <-- Swirl: u_theta ~ q^{-(1/2+h)}
                  └─────┼─────┘      Radial: u_r ~ q^{-1/2}
                        │
```

---

## 3. Key Physical & Numerical Findings

### 3.1 Energetics: Vanishing Kinetic Energy vs. Diverging Enstrophy
The core volume shrinks as $V_{\text{core}} \sim r_{\text{core}}^2 z_{\text{core}} \sim q \cdot q^{1/2 - h} = q^{3/2 - h}$.
- **Core Kinetic Energy:** 
  $$E_{\text{core}} \sim |\mathbf{u}|^2 V_{\text{core}} \sim q^{-(1 + 2h)} q^{3/2 - h} = q^{1/2 - 3h} \xrightarrow{q \to 0} 0 \quad (\text{since } 1/2 - 3h = 0.485 > 0)$$
  The kinetic energy inside the collapsing core vanishes as $q \to 0$. The total box energy is dominated by the unscaled background and remains strictly bounded.
- **Enstrophy & Beale–Kato–Majda Criterion:**
  Vorticity scales as $|\boldsymbol{\omega}| \sim |\mathbf{u}| / r_{\text{core}} \sim q^{-(1 + h)}$. Core enstrophy diverges:
  $$\Omega_{\text{core}} = \frac{1}{2} \int_{\text{core}} |\boldsymbol{\omega}|^2 \, d^3\mathbf{x} \sim q^{-(2 + 2h)} q^{3/2 - h} = q^{-(1/2 + 3h)} \approx q^{-0.515} \to \infty$$
  Because $\lVert\boldsymbol{\omega}\rVert_{L^\infty} \sim \tau^{-(1 + h)}$ and $1 + h = 1.005 > 1$, the integral $\int_0^T \lVert\boldsymbol{\omega}\rVert_{L^\infty} dt$ diverges as $\tau \to 0$: the BKM-type divergence that a blow-up of the underlying construction would require. This describes the surrogate's analytical asymptotics - the numerical runs stop at $\tau = q_{\text{floor}} > 0$ and do not integrate through the singularity.
- **Asymptotic Reynolds Number:**
  $$\mathrm{Re} _\theta = \frac{u _\theta r _{\text{core}}}{\nu} \sim \frac{q^{-(1/2 + h)} q^{1/2}}{\nu} = \frac{q^{-h}}{\nu} \xrightarrow{q \to 0} \infty$$
  Viscosity becomes asymptotically subdominant in the core, making the precursor dynamics effectively Euler-dominated.

### 3.2 The Wave-Packet Reynolds-Stress Mechanism at Discrete Resolution
The theoretical construction introduces non-axisymmetric wave packets $\mathbf{w}$ with azimuthal symmetry $m = \pm 1$ in an annular zone. Their quadratic interaction generates a Reynolds stress $\nabla \cdot \langle \mathbf{w} \otimes \mathbf{w} \rangle$ with $m = 0, 2$ harmonics designed to cancel secular growth in the base residual.

The discrete quadratic optimizer (`nsblowup/reference/quadratic.py`) fits wave amplitudes $\mathbf{a} \in \mathbb{R}^4$ to minimize the annular residual $L^2$ norm. On discrete grids ($N \le 128$), it finds:
$$\mathbf{a}^* = (0.0, \; 0.0, \; 0.0, \; 0.0)$$
**Why?** The wave carrier wavenumber scales as $k_{\text{wave}} \sim q^{-(1/2 + h/2)}$. On an $N=64$ grid with $2/3$ dealiasing ($k_{\text{cut}} = 21$), the wave wavenumber reaches the retained-band ($2/3$) cutoff by $\tau \approx 0.30$. Beyond that point, the wave packet cannot be resolved: rather than producing a smooth Reynolds stress, truncated waves inject high-gradient numerical artifacts that increase the residual. The optimizer correctly rejects them.

### 3.3 Physical Instability under Free Evolution
In the free evolution experiment (`--init-from-reference 0.9 --no-force`), the constructed state at $t = 0.9$ ($\tau = 0.10$) is released with the external force turned off ($\mathbf{f} = \mathbf{0}$).
- The collapsing core **disperses immediately**, decaying exponentially at $\frac{d \ln \|\mathbf{u}\|_{L^\infty}}{ds} \approx -3.85\,\text{s}^{-1}$.
- This demonstrates that the core is an externally driven kinematic structure, not a self-sustaining solitary attractor of the unforced Navier–Stokes equations.

---

## 4. Pipeline & Usage

```
    python cli.py reference --n 64 --calibrate --save-calibration
                            Analytical field u*(t, x); exact dual-number time
                            derivative; evaluates residual force on grid
            │
            ▼
    wave_calibration.json   Fitted wave amplitudes with configuration metadata
            │
            ▼
    python cli.py dns       Forced DNS from rest (IMEX-RK2, strict 2/3),
                            writes openai_dns.npz + openai_dns.mp4
            ├── python cli.py compare    Exponent tables & scaling plots vs theory
            └── python cli.py rerender   Re-encode movie from saved archive
```

### Tabulated Forcing Across Resolutions
To perform rigorous resolution studies where every DNS grid consumes restrictions of the exact same external forcing, generate a tabulated force on an absolute reference grid $N^*$:

```bash
python cli.py reference --n 64 --force-grid 128 --freeze-force force.npz
python cli.py dns --n 64 --force-table force.npz
```

The table stores physical force snapshots $\mathcal{P} _N \mathbf{f} _{N^{\ast}}$ on the DNS grid, a SHA-256 checksum over time nodes and tensors, and the complete configuration metadata. The time grid is piecewise: $10^{-3}$ spacing across the start-up ramp ($t < 0.06$), $5 \times 10^{-3}$ elsewhere - the ramp is the only region where a uniform grid leaves a large interpolation error.

---

## 5. Codebase Architecture

| Module | Role |
|---|---|
| `cli.py` | Single entry point dispatching all subcommands |
| [`nsblowup/core/dual.py`](nsblowup/core/dual.py) | Forward-mode automatic differentiation (dual numbers) for exact $\partial_t \mathbf{u}^*$ |
| [`nsblowup/core/spectral.py`](nsblowup/core/spectral.py) | Fourier operators, 2/3 and 3/2 dealiasing, cell-centred origin phase, spectral tail monitor |
| [`nsblowup/core/config.py`](nsblowup/core/config.py) | Dataclasses for simulation and reference settings; memory guard |
| [`nsblowup/core/frozen.py`](nsblowup/core/frozen.py) | Tabulated force loader with 4-point cubic Lagrange time interpolation |
| [`nsblowup/reference/reference.py`](nsblowup/reference/reference.py) | Analytical field construction, similarity solver, `ForceOracle` |
| [`nsblowup/reference/quadratic.py`](nsblowup/reference/quadratic.py) | Exact 15-point Gram matrix quadratic residual model and optimizer |
| [`nsblowup/reference/waves.py`](nsblowup/reference/waves.py) | Wave basis definitions, Reynolds stress moments, carrier-wavenumber gating |
| [`nsblowup/reference/report.py`](nsblowup/reference/report.py) | Momentum balance table, finite-difference cross-checks, swirl collapse diagnostics |
| [`nsblowup/reference/tables.py`](nsblowup/reference/tables.py) | Checksum fingerprints, tabulated force generation, time-interpolation checks |
| [`nsblowup/run/solver.py`](nsblowup/run/solver.py) | IMEX-RK2 pseudo-spectral solver with exact integrating factor for diffusion |
| [`nsblowup/run/volume.py`](nsblowup/run/volume.py) | Emission-absorption volumetric ray marcher and 3D tracer interpolation |
| [`nsblowup/run/video.py`](nsblowup/run/video.py) | Multi-panel diagnostic movie generator |
| [`nsblowup/run/compare.py`](nsblowup/run/compare.py) | Scaling exponent estimation and multi-resolution overlay figures |
| [`nsblowup/verify/`](nsblowup/verify/) | Method of Manufactured Solutions (MMS) test suite |

---

## 6. Execution & Verification

Requires Python $\ge$ 3.10. Install dependencies:
```bash
pip install -r requirements.txt
```

### Verification & Smoke Tests
```bash
# Method of Manufactured Solutions check (verifies solver, time stepping, Leray projector,
# and cell-centred origin phase shift to machine precision ~10^-16):
python cli.py verify

# Fast smoke test:
python cli.py dns --quick
```

### Full Experiment Suite
```bash
# 1. Reference report and wave calibration:
python cli.py reference --n 64 --calibrate --save-calibration

# 2. Baseline forced DNS:
python cli.py dns --n 64 --calibration wave_calibration.json

# 3. Post-hoc scaling analysis:
python cli.py compare

# 4. Deep collapse experiment (clock floor q_floor = 0.04):
python cli.py dns --q-floor 0.04 --t-end 0.96 --particles 6000

# 5. Free evolution experiment (release at t = 0.9 with force turned off):
python cli.py dns --init-from-reference 0.9 --no-force --t-end 1.5

# 6. Online N*=128 reference run (no table; isolates interpolation error):
python cli.py dns --force-grid 128
```

---

## 7. Numerical Techniques

- **Cell-Centred Fourier Origin Phase:** Nodes sit at $x_j = (j + 1/2)/N - 1/2$. Coarse-to-fine or fine-to-coarse grid resampling rotates Fourier modes by $\exp(-2\pi i m \delta)$ where $\delta = 1/(2N_{\text{in}}) - 1/(2N_{\text{out}})$. Without this correction, resampling shifts physical fields by half a cell; with it, restriction error is at machine precision ($4.27 \times 10^{-16}$).
- **Rotational Advective Form:** Advection is evaluated as $\mathcal{P}[(\nabla \times \mathbf{u}) \times \mathbf{u}]$, mathematically identical to $\mathcal{P}[(\mathbf{u} \cdot \nabla)\mathbf{u}]$ because $\nabla(1/2 |\mathbf{u}|^2)$ vanishes under the Leray projector, saving tensor gradient FFTs.
- **Integrating-Factor IMEX-RK2:** Stiff viscous diffusion $\nu k^2$ is integrated analytically via an exact matrix exponential, while advection and forcing are marched with a 2-stage explicit Runge–Kutta scheme with trapezoidal forcing average. Divergence remains $\le 3 \times 10^{-16}$.
- **Spectral Tail Resolution Monitor:** Monitors energy in the outer shell ($\max |m_i| > 0.8 k_{\text{cut}}$) of the dealiased band. The run halts if the tail exceeds the tolerance after the initial startup ramp.

---

## 8. Published Artifacts

Simulation archives (`.npz`), rendered videos (`.mp4`), tabulated force tables, and diagnostic reports are published as release assets:

https://github.com/CokieMiner/nsblowup/releases/tag/artifacts-v1

- `openai_dns`: Baseline forced run (700 tracers), velocity slope $-0.496 \pm 0.001$ (theory $-0.505$).
- `openai_many`: High-density visualization (6000 tracers), same trajectory.
- `openai_deep`: Deeper collapse ($q_{\text{floor}} = 0.04$, $t_{\text{end}} = 0.96$), slope $-0.489 \pm 0.001$.
- `openai_free`: Released at $t = 0.9$ with $\mathbf{f} = \mathbf{0}$, showing rapid dispersal of the released core ($-3.849 \pm 0.043\,\text{s}^{-1}$).
- `openai_table`: Driven solely by tabulated $N^* = 128$ force snapshots (piecewise time grid); completes at the standard $2 \times 10^{-3}$ tail tolerance with pre-arm peak $1.87 \times 10^{-3}$, identical to the online and baseline runs.
- `openai_online`: Same forcing evaluated online on $N^* = 128$ with no table; it reaches the same pre-arm peak as the table run, and the piecewise grid keeps the ramp interpolation error at $3.6 \times 10^{-5}$ - the frozen table adds no measurable start-up transient.

---

## 9. License & References

- License: Apache-2.0 (see [`LICENSE`](LICENSE)).
- Sources: [OpenAI Navier–Stokes Announcement](https://openai.com/index/navier-stokes-solution/), [Lean Formalization](https://github.com/openai/NavierStokesAndEuler), and the [Clay Mathematics Millennium Problem](https://www.claymath.org/millennium-problems/navier-stokes-equation/).
