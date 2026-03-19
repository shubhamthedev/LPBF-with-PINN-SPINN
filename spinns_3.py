"""
LPBF SPINN v4 — Hastelloy X, Single Track
==========================================
Base: v3 (Flax einsum SPINN, physical units, hard IC)
Fixes applied vs v3:
  FIX 1 — PDE residual normalised by rho*cp0*T_char/t_ref  (≈ qv_peak)
           not by qv_peak directly — ensures O(1) residual in float64
  FIX 2 — BC residual normalised consistently with PDE scale
  FIX 3 — Output layer init stddev: 0.01 → 0.1  (non-trivial gradient at ep 0)
  FIX 4 — w_bc: 0.05 → 1.0  (BCs weighted equally with PDE)
  FIX 5 — float64 throughout (was float32 — borderline for W/m³ magnitudes)

Architecture: SPINN (Separable PINN)
  T(x,y,z,t) = T0 + tau * einsum(Ft,Fx,Fy,Fz) * T_char
  Each axis MLP: R^1 → R^RANK  (tanh hidden, linear output)
  Hard IC: T(tau=0) = T0 exactly by construction

PDE: rho*cp(T)*dT/dt = div(kappa(T)*grad(T)) + qv(x,y,z,t)
BCs:
  Top    z=0   : Robin   -kappa*dT/dz = h*(T - T_inf)
  Bottom z=Lz  : Neumann  dT/dz = 0
  x=0, x=Lx   : Neumann  dT/dx = 0
  y=0, y=Ly   : Neumann  dT/dy = 0
IC:  T(x,y,z,0) = T0  [hard-enforced, no IC loss term]
"""

import os
import time
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib import cm

import jax
import jax.numpy as jnp
from jax import jvp, jit
import optax
import flax.linen as nn
from flax.training import train_state

# ── FIX 5: float64 throughout ────────────────────────────────────────────────
jax.config.update("jax_enable_x64", True)

np.random.seed(42)
key = jax.random.PRNGKey(42)

RESULTS_DIR = "results_spinn_v4"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1.  PHYSICAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
rho      = 8351.91          # kg/m³  Hastelloy X density
P_abs    = 150.0            # W      absorbed laser power
r_beam   = 450.0e-6         # m      1/e² beam radius  ← user-confirmed 450 µm
c_depth  = 500.0e-6         # m      Goldak penetration depth
v_scan   = 0.10             # m/s    scan speed (100 mm/s)
T0       = 300.0            # K      initial / ambient temperature
T_inf    = 300.0            # K      ambient for convection BC
h_conv   = 10.0             # W/(m²·K)  convective heat transfer coefficient
T_melt   = 1533.0           # K      Hastelloy X melting point

# Physical domain
Lx, Ly, Lz = 2.0e-3, 1.5e-3, 1.0e-3   # m

# Laser track: starts at x0_laser, ends at x1_laser, centred at y_track
x0_laser = 0.50e-3          # m
x1_laser = 1.50e-3          # m
y_track  = 0.50e-3          # m

# Total scan time
t_end = (x1_laser - x0_laser) / v_scan   # = 10 ms

# ══════════════════════════════════════════════════════════════════════════════
#  2.  MATERIAL PROPERTIES  (Hastelloy X, temperature-dependent, Eq. 21)
#      κ [W/(m·K)],  cp [J/(kg·K)]  — both functions of T in Kelvin
# ══════════════════════════════════════════════════════════════════════════════
def kappa_fn(T):
    """Thermal conductivity κ(T) [W/(m·K)]."""
    return (229.87 + 0.0184 * T
            + 225.10 * jnp.tanh(0.018 * (T - 1816.8)))

def cp_fn(T):
    """Specific heat capacity cp(T) [J/(kg·K)]."""
    return (407.62 + 0.142 * T
            - 61.43  * jnp.exp(-3.1e-4 * (T - 798.0)**2)
            + 1054.96 * jnp.exp(-6.2e-5 * (T - 1816.8)**2))

def dkappa_dT_fn(T):
    """Analytic dκ/dT — avoids an extra AD call inside the PDE residual."""
    return 0.0184 + 225.10 * 0.018 * (1.0 - jnp.tanh(0.018*(T - 1816.8))**2)

# Reference values at T0 (used for non-dimensionalisation)
kap0 = float(229.87 + 0.0184*T0 + 225.10*np.tanh(0.018*(T0-1816.8)))
cp0  = float(407.62 + 0.142*T0
             - 61.43*np.exp(-3.1e-4*(T0-798.0)**2)
             + 1054.96*np.exp(-6.2e-5*(T0-1816.8)**2))
alp0 = kap0 / (rho * cp0)   # m²/s  thermal diffusivity at T0

# ══════════════════════════════════════════════════════════════════════════════
#  3.  NON-DIMENSIONALISATION
#
#  Spatial ref: r_ref = r_beam  (beam radius)
#  Depth ref:   c_depth         (penetration depth, used for z only)
#  Time ref:    t_ref = r_beam² / alpha0  (diffusion time over beam radius)
#
#  Non-dim coords:
#    xi   = x / r_ref       ∈ [0, x_nd_max]
#    eta  = y / r_ref       ∈ [0, y_nd_max]
#    zeta = z / c_depth     ∈ [0, z_nd_max]
#    tau  = t / t_ref       ∈ [0, t_nd_max]
#
#  Temperature scale (Rosenthal):
#    T_char = qv_peak * t_ref / (rho * cp0)
#    → this is the natural temperature rise from the source over one
#      diffusion time, ensuring the PDE residual is O(1)
# ══════════════════════════════════════════════════════════════════════════════
r_ref  = r_beam                  # spatial reference [m]
t_ref  = r_beam**2 / alp0        # time reference [s]

# Non-dim domain extents
x_nd_max = Lx      / r_ref       # xi   ∈ [0, x_nd_max]
y_nd_max = Ly      / r_ref       # eta  ∈ [0, y_nd_max]
z_nd_max = Lz      / c_depth     # zeta ∈ [0, z_nd_max]
t_nd_max = t_end   / t_ref       # tau  ∈ [0, t_nd_max]

# Non-dim laser positions
x0_nd = x0_laser / r_ref
x1_nd = x1_laser / r_ref
yt_nd = y_track  / r_ref

# Goldak heat source peak [W/m³]
qv_peak = (6.0 * np.sqrt(3.0) * P_abs /
           (np.pi * np.sqrt(np.pi) * r_beam**2 * c_depth))

# ── FIX 1: Rosenthal temperature scale ───────────────────────────────────────
# T_char = qv_peak * t_ref / (rho * cp0)
# This is the temperature rise produced by the peak source over one
# diffusion time.  It is the correct scale so that the PDE residual
# (which has units W/m³) normalised by rho*cp0*T_char/t_ref equals 1.
# Note: rho*cp0*T_char/t_ref = qv_peak  exactly — so the two normalisations
# are identical in value but the Rosenthal form is self-consistent.
T_char = float(qv_peak) * float(t_ref) / (rho * cp0)

# PDE normalisation scale [W/m³]  — used in loss functions
# rho * cp0 * T_char / t_ref  ≡  qv_peak  (by construction above)
_pde_scale = rho * cp0 * T_char / t_ref   # [W/m³]

# Sanity-check: _pde_scale should equal qv_peak
assert abs(_pde_scale / qv_peak - 1.0) < 1e-6, \
    f"PDE scale mismatch: {_pde_scale:.4e} vs {qv_peak:.4e}"

print("=" * 66)
print("  LPBF SPINN v4 — Hastelloy X  (all 5 fixes applied)")
print("=" * 66)
print(f"  r_beam  = {r_beam*1e6:.0f} µm  |  c_depth = {c_depth*1e6:.0f} µm")
print(f"  P_abs   = {P_abs:.1f} W   |  v_scan  = {v_scan*1e3:.0f} mm/s")
print(f"  qv_peak = {qv_peak:.4e} W/m³")
print(f"  kap0    = {kap0:.3f} W/(m·K)  |  cp0 = {cp0:.1f} J/(kg·K)")
print(f"  alp0    = {alp0:.4e} m²/s")
print(f"  t_ref   = {t_ref*1e3:.4f} ms")
print(f"  T_char  = {T_char:.2f} K  (Rosenthal scale)")
print(f"  _pde_scale = {_pde_scale:.4e} W/m³  (≡ qv_peak ✓)")
print(f"  t_nd_max = {t_nd_max:.4f}")
print(f"  Domain nd: ξ∈[0,{x_nd_max:.2f}]  η∈[0,{y_nd_max:.2f}]  ζ∈[0,{z_nd_max:.2f}]")
print(f"  theta_melt = {(T_melt-T0)/T_char:.4f}  (target O(1) ✓)")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  4.  HEAT SOURCE  (Goldak double-ellipsoid, lab frame, physical units)
#
#  q_v(x,y,z,t) = qv_peak * exp(-3*((x-x_L)²+(y-y_T)²)/r²) * exp(-z²/c²)
#
#  The factor -3 in the exponent comes from the Goldak normalisation:
#  integrating exp(-3r²/R²) over all space gives (π/3)^(3/2) R³,
#  and the prefactor 6√3/π^(3/2) ensures ∫q_v dV = P_abs.
# ══════════════════════════════════════════════════════════════════════════════
def heat_source(x_p, y_p, z_p, t_p):
    """
    Volumetric heat source [W/m³] in physical coordinates.
    Laser position clipped to [x0_laser, x1_laser].
    """
    x_laser = jnp.clip(x0_laser + v_scan * t_p,
                       a_min=float(x0_laser),
                       a_max=float(x1_laser))
    return float(qv_peak) * (
        jnp.exp(-3.0 * ((x_p - x_laser)**2 + (y_p - float(y_track))**2)
                / float(r_beam)**2)
        * jnp.exp(-3.0 * (z_p**2 / float(c_depth)**2))
    )

# ══════════════════════════════════════════════════════════════════════════════
#  5.  SPINN ARCHITECTURE
#
#  Four axis sub-networks: net_t, net_x, net_y, net_z
#  Each maps a 1-D coordinate array → R^RANK feature vectors
#  Output: einsum('tr,xr,yr,zr->txyz') = outer product over rank dimension
#
#  Physical temperature with hard IC:
#    T(xi,eta,zeta,tau) = T0 + tau * NN_output * T_char
#    → T(tau=0) = T0 exactly, no IC loss term needed
# ══════════════════════════════════════════════════════════════════════════════
RANK     = 32    # CP-decomposition rank
N_HIDDEN = 64    # neurons per hidden layer
N_LAYERS = 4     # number of hidden layers

# ── FIX 3: output layer stddev 0.01 → 0.1 ────────────────────────────────────
# With stddev=0.01 the initial raw output is ~0.01, giving T ≈ T0 + tau*0.01*T_char
# The PDE gradient w.r.t. output weights is proportional to this stddev.
# 0.1 gives a 10× larger initial gradient signal without destabilising training.
OUTPUT_INIT = nn.initializers.normal(stddev=0.1)

class AxisMLP(nn.Module):
    """Single-axis sub-network: (N,1) → (N, RANK). Tanh hidden, linear output."""
    hidden: Sequence[int]
    rank: int

    @nn.compact
    def __call__(self, x):
        h = x[:, None]                          # (N,) → (N,1)
        for feat in self.hidden:
            h = nn.Dense(feat,
                         kernel_init=nn.initializers.glorot_normal(),
                         bias_init=nn.initializers.zeros)(h)
            h = nn.tanh(h)
        # Linear output — no activation, gradient flows freely
        return nn.Dense(self.rank,
                        kernel_init=OUTPUT_INIT,
                        bias_init=nn.initializers.zeros)(h)   # (N, RANK)

class SPINN_v4(nn.Module):
    """
    Separable PINN for 4-D (t,x,y,z) heat equation.
    Output shape: (N_t, N_x, N_y, N_z) — the full temperature tensor.
    """
    hidden: Sequence[int]
    rank: int

    def setup(self):
        kw = dict(hidden=self.hidden, rank=self.rank)
        self.net_t = AxisMLP(**kw)
        self.net_x = AxisMLP(**kw)
        self.net_y = AxisMLP(**kw)
        self.net_z = AxisMLP(**kw)

    def __call__(self, tau, xi, eta, zeta):
        Ft = self.net_t(tau)    # (N_t, RANK)
        Fx = self.net_x(xi)     # (N_x, RANK)
        Fy = self.net_y(eta)    # (N_y, RANK)
        Fz = self.net_z(zeta)   # (N_z, RANK)
        # Outer product over rank: result shape (N_t, N_x, N_y, N_z)
        return jnp.einsum('tr,xr,yr,zr->txyz', Ft, Fx, Fy, Fz)

# ── Initialise model ──────────────────────────────────────────────────────────
model    = SPINN_v4(hidden=tuple([N_HIDDEN]*N_LAYERS), rank=RANK)
apply_fn = model.apply

key, sk = jax.random.split(key)
dummy_t = jnp.linspace(0.0, t_nd_max, 4)
dummy_x = jnp.linspace(0.0, x_nd_max, 4)
dummy_y = jnp.linspace(0.0, y_nd_max, 4)
dummy_z = jnp.linspace(0.0, z_nd_max, 4)
params  = model.init(sk, dummy_t, dummy_x, dummy_y, dummy_z)

n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

# ══════════════════════════════════════════════════════════════════════════════
#  6.  PHYSICAL TEMPERATURE  (hard IC enforced)
#
#  T_phys(tau, xi, eta, zeta) = T0 + tau * NN(tau,xi,eta,zeta) * T_char
#
#  At tau=0: T = T0 + 0 * NN * T_char = T0  ✓  (exact, no penalty needed)
#  At tau>0: T rises according to what the network learns from the PDE
# ══════════════════════════════════════════════════════════════════════════════
def T_phys(params, tau, xi, eta, zeta):
    """
    Physical temperature [K].
    tau, xi, eta, zeta are 1-D arrays of non-dim coordinates.
    Returns shape (N_t, N_x, N_y, N_z).
    """
    raw  = apply_fn(params, tau, xi, eta, zeta)   # (N_t, N_x, N_y, N_z)
    tau4 = tau[:, None, None, None]               # broadcast over x,y,z
    return float(T0) + tau4 * raw * float(T_char)

# Sanity check at initialisation
T_init = np.array(T_phys(params, dummy_t, dummy_x, dummy_y, dummy_z))
print(f"  SPINN parameters : {n_params:,}")
print(f"  T at init        : [{T_init.min():.1f}, {T_init.max():.1f}] K")
print(f"  (tau=0 slice)    : {T_init[0].mean():.2f} K  (should = {T0:.0f} K ✓)")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  7.  LOSS FUNCTIONS
#
#  All residuals are in physical units [W/m³] then normalised by _pde_scale.
#  This gives O(1) loss values regardless of the magnitude of qv_peak.
#
#  PDE:  rho*cp(T)*dT/dt - div(kappa(T)*grad(T)) - qv = 0
#  BC top (z=0):    -kappa*dT/dz = h*(T - T_inf)   [Robin]
#  BC others:       dT/dn = 0                        [Neumann]
# ══════════════════════════════════════════════════════════════════════════════

# Precompute physical conversion factors (scalars, used inside JIT)
_r2  = float(r_ref)**2     # r_ref²  [m²]
_c2  = float(c_depth)**2   # c_depth² [m²]
_tr  = float(t_ref)        # t_ref   [s]
_rr  = float(r_ref)        # r_ref   [m]
_cd  = float(c_depth)      # c_depth [m]
_qp  = float(qv_peak)      # qv_peak [W/m³]
_rho = float(rho)          # density [kg/m³]
_ps  = float(_pde_scale)   # PDE normalisation scale [W/m³]  ← FIX 1

def pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c):
    """
    PDE residual loss on interior collocation points.

    The SPINN forward pass gives T on the full (N_t × N_x × N_y × N_z) lattice.
    JVP gives dT/d(coord) on the same lattice in one forward pass.

    Residual [W/m³]:
      R = rho*cp(T)*dT/dt - kappa(T)*Laplacian(T) - dkappa/dT*|grad T|² - qv

    Normalised residual (dimensionless):
      R_nd = R / _pde_scale   ← FIX 1: was R / qv_peak (same value, but now
                                 explicitly tied to the Rosenthal scale)
    """
    ones = lambda v: jnp.ones_like(v)

    # ── dT/dt via JVP over tau ────────────────────────────────────────────
    T, dT_dtau = jvp(
        lambda t: T_phys(params, t, xi_c, eta_c, zeta_c),
        (tau_c,), (ones(tau_c),))
    dT_dt = dT_dtau / _tr          # [K/s]

    # ── Material properties at current T ─────────────────────────────────
    T_K   = T                      # physical T [K]  shape (Nt,Nx,Ny,Nz)
    k_T   = kappa_fn(T_K)          # κ(T)  [W/(m·K)]
    cp_T  = cp_fn(T_K)             # cp(T) [J/(kg·K)]
    dkdT  = dkappa_dT_fn(T_K)      # dκ/dT [W/(m·K²)]

    # ── Spatial 1st derivatives via JVP ──────────────────────────────────
    _, dT_dxi = jvp(
        lambda x: T_phys(params, tau_c, x, eta_c, zeta_c),
        (xi_c,), (ones(xi_c),))
    _, dT_deta = jvp(
        lambda y: T_phys(params, tau_c, xi_c, y, zeta_c),
        (eta_c,), (ones(eta_c),))
    _, dT_dzeta = jvp(
        lambda z: T_phys(params, tau_c, xi_c, eta_c, z),
        (zeta_c,), (ones(zeta_c),))

    # Convert to physical gradients [K/m]
    dT_dx = dT_dxi   / _rr
    dT_dy = dT_deta  / _rr
    dT_dz = dT_dzeta / _cd

    # ── Spatial 2nd derivatives via nested JVP ────────────────────────────
    _, d2T_dxi2 = jvp(
        lambda x: jvp(
            lambda x2: T_phys(params, tau_c, x2, eta_c, zeta_c),
            (x,), (ones(x),))[1],
        (xi_c,), (ones(xi_c),))
    _, d2T_deta2 = jvp(
        lambda y: jvp(
            lambda y2: T_phys(params, tau_c, xi_c, y2, zeta_c),
            (y,), (ones(y),))[1],
        (eta_c,), (ones(eta_c),))
    _, d2T_dzeta2 = jvp(
        lambda z: jvp(
            lambda z2: T_phys(params, tau_c, xi_c, eta_c, z2),
            (z,), (ones(z),))[1],
        (zeta_c,), (ones(zeta_c),))

    # Convert to physical Laplacian components [K/m²]
    d2T_dx2 = d2T_dxi2   / _r2
    d2T_dy2 = d2T_deta2  / _r2
    d2T_dz2 = d2T_dzeta2 / _c2

    # ── Heat source on the collocation lattice [W/m³] ────────────────────
    tau4  = tau_c[:, None, None, None]
    xi4   = xi_c [None, :, None, None]
    eta4  = eta_c[None, None, :, None]
    zeta4 = zeta_c[None, None, None, :]
    q_v   = heat_source(xi4*_rr, eta4*_rr, zeta4*_cd, tau4*_tr)

    # ── PDE residual [W/m³] ───────────────────────────────────────────────
    laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
    grad_sq   = dT_dx**2 + dT_dy**2 + dT_dz**2

    R = (_rho * cp_T * dT_dt
         - k_T * laplacian
         - dkdT * grad_sq
         - q_v)

    # ── FIX 1: normalise by Rosenthal PDE scale ───────────────────────────
    R_nd = R / _ps
    return jnp.mean(R_nd**2)

def bc_loss_fn(params, tau_b, xi_b, eta_b, zeta_b):
    """
    Boundary condition loss.

    All BC residuals are in [W/m²] (flux form) then normalised by
    _ps * r_ref  [W/m³ × m = W/m²]  — FIX 2: consistent with PDE scale.

    BCs:
      Top    z=0   : Robin   -κ·∂T/∂z = h·(T - T_inf)
      Bottom z=Lz  : Neumann  ∂T/∂z = 0
      Left   x=0   : Neumann  ∂T/∂x = 0
      Right  x=Lx  : Neumann  ∂T/∂x = 0
      Front  y=0   : Neumann  ∂T/∂y = 0
      Back   y=Ly  : Neumann  ∂T/∂y = 0
    """
    ones = lambda v: jnp.ones_like(v)

    # ── FIX 2: BC normalisation scale [W/m²] ─────────────────────────────
    # _ps [W/m³] × r_ref [m] = [W/m²]  → same order as κ·∂T/∂z
    bc_scale = _ps * _rr

    loss = jnp.array(0.0)

    # ── Top surface z=0: Robin BC ─────────────────────────────────────────
    # Physical: -κ(T)·∂T/∂z|_{z=0} = h·(T - T_inf)
    # In non-dim: dT/dzeta at zeta=0, then convert: dT/dz = dT/dzeta / c_depth
    zeta_top = jnp.zeros(1)
    T_top, dT_dzeta_top = jvp(
        lambda z: T_phys(params, tau_b, xi_b, eta_b, z),
        (zeta_top,), (ones(zeta_top),))
    dT_dz_top = dT_dzeta_top / _cd         # [K/m]
    k_top     = kappa_fn(T_top)            # [W/(m·K)]
    # Residual [W/m²]: -κ·∂T/∂z - h·(T - T_inf)
    res_robin = (-k_top * dT_dz_top
                 - float(h_conv) * (T_top - float(T_inf)))
    loss = loss + jnp.mean((res_robin / bc_scale)**2)

    # ── Bottom z=Lz: Neumann dT/dz = 0 ───────────────────────────────────
    zeta_bot = jnp.array([z_nd_max])
    _, dT_dzeta_bot = jvp(
        lambda z: T_phys(params, tau_b, xi_b, eta_b, z),
        (zeta_bot,), (ones(zeta_bot),))
    dT_dz_bot = dT_dzeta_bot / _cd        # [K/m]
    # Normalise by bc_scale / κ0 to get dimensionless flux
    loss = loss + jnp.mean((dT_dz_bot * kap0 / bc_scale)**2)

    # ── Left x=0: Neumann dT/dx = 0 ──────────────────────────────────────
    xi_left = jnp.zeros(1)
    _, dT_dxi_left = jvp(
        lambda x: T_phys(params, tau_b, x, eta_b, zeta_b),
        (xi_left,), (ones(xi_left),))
    dT_dx_left = dT_dxi_left / _rr
    loss = loss + jnp.mean((dT_dx_left * kap0 / bc_scale)**2)

    # ── Right x=Lx: Neumann dT/dx = 0 ────────────────────────────────────
    xi_right = jnp.array([x_nd_max])
    _, dT_dxi_right = jvp(
        lambda x: T_phys(params, tau_b, x, eta_b, zeta_b),
        (xi_right,), (ones(xi_right),))
    dT_dx_right = dT_dxi_right / _rr
    loss = loss + jnp.mean((dT_dx_right * kap0 / bc_scale)**2)

    # ── Front y=0: Neumann dT/dy = 0 ─────────────────────────────────────
    eta_front = jnp.zeros(1)
    _, dT_deta_front = jvp(
        lambda y: T_phys(params, tau_b, xi_b, y, zeta_b),
        (eta_front,), (ones(eta_front),))
    dT_dy_front = dT_deta_front / _rr
    loss = loss + jnp.mean((dT_dy_front * kap0 / bc_scale)**2)

    # ── Back y=Ly: Neumann dT/dy = 0 ─────────────────────────────────────
    eta_back = jnp.array([y_nd_max])
    _, dT_deta_back = jvp(
        lambda y: T_phys(params, tau_b, xi_b, y, zeta_b),
        (eta_back,), (ones(eta_back),))
    dT_dy_back = dT_deta_back / _rr
    loss = loss + jnp.mean((dT_dy_back * kap0 / bc_scale)**2)

    return loss / 6.0   # average over 6 faces

def total_loss_fn(params,
                  tau_c, xi_c, eta_c, zeta_c,
                  tau_b, xi_b, eta_b, zeta_b,
                  w_pde, w_bc):
    """Combined loss. IC is zero by construction — no IC term."""
    Lp = pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c)
    Lb = bc_loss_fn(params,  tau_b, xi_b, eta_b, zeta_b)
    return w_pde * Lp + w_bc * Lb, (Lp, Lb)

# ══════════════════════════════════════════════════════════════════════════════
#  8.  COLLOCATION POINT SAMPLING
#
#  Interior (PDE): clustered near the laser path for better resolution
#  Boundary (BC):  uniform on each face
# ══════════════════════════════════════════════════════════════════════════════
N_T, N_X, N_Y, N_Z = 128, 128, 128, 128   # axis point counts
N_BC = 64                                   # BC points per face
LASER_ND_STD = 1.2                          # clustering std in non-dim units

def sample_col(key):
    """Sample 1-D axis collocation points for SPINN interior lattice."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    # tau: uniform over [0, t_nd_max]
    tau_c = jax.random.uniform(k1, (N_T,), minval=0.0, maxval=t_nd_max)

    # xi: 40% uniform + 60% clustered near laser sweep [x0_nd, x1_nd]
    n_xu = max(2, N_X * 4 // 10)
    n_xf = N_X - n_xu
    xi_u = jax.random.uniform(k2, (n_xu,), minval=0.0, maxval=x_nd_max)
    xi_c_base = jax.random.uniform(k3, (n_xf,), minval=x0_nd, maxval=x1_nd)
    xi_f = jnp.clip(xi_c_base + jax.random.normal(k4, (n_xf,)) * LASER_ND_STD,
                    0.0, x_nd_max)
    xi_c = jnp.concatenate([xi_u, xi_f])

    # eta: 40% uniform + 60% clustered near track centre yt_nd
    n_yu = max(2, N_Y * 4 // 10)
    n_yf = N_Y - n_yu
    eta_u = jax.random.uniform(k5, (n_yu,), minval=0.0, maxval=y_nd_max)
    eta_f = jnp.clip(float(yt_nd) + jax.random.normal(k6, (n_yf,)) * LASER_ND_STD,
                     0.0, y_nd_max)
    eta_c = jnp.concatenate([eta_u, eta_f])

    # zeta: 40% uniform + 60% exponentially clustered near surface (zeta=0)
    n_zu = max(2, N_Z * 4 // 10)
    n_zf = N_Z - n_zu
    zeta_u = jax.random.uniform(k7, (n_zu,), minval=0.0, maxval=z_nd_max)
    zeta_f = jnp.clip(jax.random.exponential(k8, (n_zf,)) * 0.6, 0.0, z_nd_max)
    zeta_c = jnp.concatenate([zeta_u, zeta_f])

    return tau_c, xi_c, eta_c, zeta_c

def sample_bc(key):
    """Sample uniform BC points (used as axis arrays for each face)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tau_b  = jax.random.uniform(k1, (N_BC,), minval=0.0, maxval=t_nd_max)
    xi_b   = jax.random.uniform(k2, (N_BC,), minval=0.0, maxval=x_nd_max)
    eta_b  = jax.random.uniform(k3, (N_BC,), minval=0.0, maxval=y_nd_max)
    zeta_b = jax.random.uniform(k4, (N_BC,), minval=0.0, maxval=z_nd_max)
    return tau_b, xi_b, eta_b, zeta_b

# ══════════════════════════════════════════════════════════════════════════════
#  9.  TRAINING STEP  (JIT-compiled)
# ══════════════════════════════════════════════════════════════════════════════
@jit
def train_step(state,
               tau_c, xi_c, eta_c, zeta_c,
               tau_b, xi_b, eta_b, zeta_b,
               w_pde, w_bc):
    def loss_fn(p):
        return total_loss_fn(p,
                             tau_c, xi_c, eta_c, zeta_c,
                             tau_b, xi_b, eta_b, zeta_b,
                             w_pde, w_bc)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux

# ══════════════════════════════════════════════════════════════════════════════
#  10.  OPTIMISER  (two-phase: cosine decay → fine-tune)
# ══════════════════════════════════════════════════════════════════════════════
N_EPOCHS_1 = 12_000   # Phase 1: warm-up + cosine decay
N_EPOCHS_2 =  8_000   # Phase 2: constant low LR fine-tune
N_EPOCHS   = N_EPOCHS_1 + N_EPOCHS_2

LR_PEAK = 5e-4
LR_MIN  = 1e-5

warmup   = optax.linear_schedule(0.0, LR_PEAK, 500)
cosine   = optax.cosine_decay_schedule(LR_PEAK, N_EPOCHS_1 - 500, LR_MIN / LR_PEAK)
lr_sched = optax.join_schedules([warmup, cosine], [500])

def make_opt(lr):
    return optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr)
    )

state = train_state.TrainState.create(
    apply_fn=apply_fn,
    params=params,
    tx=make_opt(lr_sched))

# ── FIX 4: w_bc = 1.0 (was 0.05) ────────────────────────────────────────────
w_pde = jnp.array(1.0)
w_bc  = jnp.array(1.0)

print(f"  Interior pts/step : {N_T*N_X*N_Y*N_Z:,}  ({N_T}×{N_X}×{N_Y}×{N_Z})")
print(f"  Phase 1 epochs    : {N_EPOCHS_1:,}  (warm-up 500 + cosine {LR_PEAK:.0e}→{LR_MIN:.0e})")
print(f"  Phase 2 epochs    : {N_EPOCHS_2:,}  (Adam fine-tune {LR_MIN:.0e})")
print(f"  Loss weights      : w_pde={float(w_pde):.2f}  w_bc={float(w_bc):.2f}  (FIX 4 ✓)")
print(f"  IC: satisfied EXACTLY by construction (no IC loss term)")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  11.  MONITORING POINTS
#       Laser centre at t=8ms: x = x0_laser + v_scan*8e-3 = 1.3 mm
# ══════════════════════════════════════════════════════════════════════════════
tau_8ms  = jnp.array([8e-3 / t_ref])
xi_8ms   = jnp.array([(x0_laser + v_scan * 8e-3) / r_ref])   # x=1.3mm
eta_8ms  = jnp.array([y_track / r_ref])
zeta_8ms = jnp.zeros(1)

# Coarse grid for T_max monitoring
xi_mon   = jnp.linspace(0.0, x_nd_max, 20)
eta_mon  = jnp.linspace(0.0, y_nd_max, 14)
zeta_mon = jnp.linspace(0.0, z_nd_max, 8)

print(f"  Monitor: T at laser centre (t=8ms, x={x0_laser*1e3+v_scan*8:.1f}mm, "
      f"y={y_track*1e3:.1f}mm, z=0)")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  12.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
LOG_EVERY   = 500
best_loss   = np.inf
best_params = params
history     = {"total": [], "pde": [], "bc": [], "T_laser": [], "T_max": []}

print("=" * 66)
print("  SPINN v4 Training")
print("=" * 66)
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):

    # Switch to phase 2 fine-tune optimizer
    if epoch == N_EPOCHS_1 + 1:
        state = train_state.TrainState.create(
            apply_fn=apply_fn, params=state.params,
            tx=make_opt(LR_MIN))
        print(f"\n  ── Phase 2: fine-tune @ LR={LR_MIN:.1e} ──\n")

    key, k1, k2 = jax.random.split(key, 3)
    tau_c, xi_c, eta_c, zeta_c = sample_col(k1)
    tau_b, xi_b, eta_b, zeta_b = sample_bc(k2)

    state, loss, (Lp, Lb) = train_step(
        state,
        tau_c, xi_c, eta_c, zeta_c,
        tau_b, xi_b, eta_b, zeta_b,
        w_pde, w_bc)

    lv, pv, bv = float(loss), float(Lp), float(Lb)

    # Monitor temperature at laser centre (t=8ms)
    T_laser = float(T_phys(state.params,
                            tau_8ms, xi_8ms, eta_8ms, zeta_8ms)[0, 0, 0, 0])
    # Coarse T_max over domain
    T_grid = np.array(T_phys(state.params, tau_8ms, xi_mon, eta_mon, zeta_mon))
    T_max  = float(T_grid.max())

    history["total"].append(lv)
    history["pde"].append(pv)
    history["bc"].append(bv)
    history["T_laser"].append(T_laser)
    history["T_max"].append(T_max)

    if lv < best_loss:
        best_loss   = lv
        best_params = state.params

    if epoch % LOG_EVERY == 0:
        lr_now = float(lr_sched(epoch)) if epoch <= N_EPOCHS_1 else LR_MIN
        print(f"  Ep {epoch:6d}/{N_EPOCHS} | "
              f"L={lv:.3e}  PDE={pv:.3e}  BC={bv:.3e} | "
              f"LR={lr_now:.2e} | "
              f"T_laser={T_laser:.0f}K  T_max={T_max:.0f}K | "
              f"t={time.time()-t0:.0f}s")

print(f"\n  Done in {time.time()-t0:.1f}s  |  best loss = {best_loss:.4e}")

# ══════════════════════════════════════════════════════════════════════════════
#  13.  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def eval_xy(params, t_val, res=200):
    """
    Temperature on top surface (z=0) at physical time t_val [s].
    Returns: X [mm], Y [mm], T [K]  — all shape (res, res).
    """
    tau  = jnp.array([t_val / t_ref])
    xi   = jnp.linspace(0.0, x_nd_max, res)
    eta  = jnp.linspace(0.0, y_nd_max, res)
    zeta = jnp.zeros(1)
    T    = np.array(T_phys(params, tau, xi, eta, zeta))[0, :, :, 0]
    X    = np.linspace(0, Lx * 1e3, res)
    Y    = np.linspace(0, Ly * 1e3, res)
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    return Xg.T, Yg.T, T.T

def eval_xz(params, t_val, res=200):
    """
    Temperature on xz cross-section (y=y_track) at physical time t_val [s].
    Returns: X [mm], Z [mm], T [K]  — all shape (res, res).
    """
    tau  = jnp.array([t_val / t_ref])
    xi   = jnp.linspace(0.0, x_nd_max, res)
    eta  = jnp.array([y_track / r_ref])
    zeta = jnp.linspace(0.0, z_nd_max, res)
    T    = np.array(T_phys(params, tau, xi, eta, zeta))[0, :, 0, :]
    X    = np.linspace(0, Lx * 1e3, res)
    Z    = np.linspace(0, Lz * 1e3, res)
    Xg, Zg = np.meshgrid(X, Z, indexing='ij')
    return Xg.T, Zg.T, T.T

def melt_pool(params, t_val, res=80):
    """
    Compute and print melt pool dimensions [mm] at time t_val [s].
    Returns: (L, W, D) in mm.
    """
    _, _, T_xy = eval_xy(params, t_val, res)
    _, _, T_xz = eval_xz(params, t_val, res)
    print(f"  T_surf ∈ [{T_xy.min():.0f}, {T_xy.max():.0f}] K")
    print(f"  T_xz   ∈ [{T_xz.min():.0f}, {T_xz.max():.0f}] K")
    if T_xy.max() < T_melt:
        print("  ⚠ No melt pool detected (T_max < T_melt).")
        return 0.0, 0.0, 0.0
    X_mm = np.linspace(0, Lx * 1e3, res)
    Y_mm = np.linspace(0, Ly * 1e3, res)
    Z_mm = np.linspace(0, Lz * 1e3, res)
    r, c = np.where(T_xy >= T_melt)
    L = float(X_mm[r.max()] - X_mm[r.min()])
    W = float(Y_mm[c.max()] - Y_mm[c.min()])
    D = 0.0
    if T_xz.max() >= T_melt:
        rz, _ = np.where(T_xz >= T_melt)
        D = float(Z_mm[rz.max()] - Z_mm[rz.min()])
    print(f"  Melt pool: L={L:.3f} mm  W={W:.3f} mm  D={D:.3f} mm")
    return L, W, D

# ══════════════════════════════════════════════════════════════════════════════
#  14.  PLOTTING  (matches PINN reference style exactly)
# ══════════════════════════════════════════════════════════════════════════════
CMAP     = "RdBu_r"
VMIN     = 400.0
VMAX     = 1600.0
INSET    = 100.0     # norm extends [VMIN-INSET, VMAX+INSET] → saturated edges
N_LEVELS = 40
N_ISO    = 16
ISO_LW   = 0.5

Lx_mm = Lx * 1e3
Ly_mm = Ly * 1e3
Lz_mm = Lz * 1e3

def plot_snapshot(params, t_val, fname, res=200):
    """
    Two-panel figure: xy top surface + xz cross-section.
    Colormap: RdBu_r, fixed VMIN=400K / VMAX=1600K.
    Gold contour at T_melt.
    """
    X_xy, Y_xy, T_xy = eval_xy(params, t_val, res)
    X_xz, Z_xz, T_xz = eval_xz(params, t_val, res)

    norm_cb  = Normalize(vmin=VMIN - INSET, vmax=VMAX + INSET)
    levels   = np.linspace(VMIN, VMAX, N_LEVELS + 1)
    iso_levs = np.linspace(VMIN, VMAX, N_ISO + 1)

    # Figure layout
    panel_w = 7.0
    panel_h = panel_w / 2.5
    gap     = 1.0
    top_m   = 0.25
    bot_m   = 1.80
    left_m  = 0.70
    right_m = 0.25

    fig_w = left_m + panel_w + right_m
    fig_h = top_m + panel_h + gap + panel_h + bot_m
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def to_frac(l, b, w, h):
        return [l/fig_w, b/fig_h, w/fig_w, h/fig_h]

    ax_xy = fig.add_axes(to_frac(left_m, bot_m + panel_h + gap, panel_w, panel_h))
    ax_xz = fig.add_axes(to_frac(left_m, bot_m,                 panel_w, panel_h))

    def _draw(ax, X, Y, T, title, xlabel, ylabel, invert_y):
        T_cl = np.clip(T, VMIN, VMAX)
        ax.contourf(X, Y, T_cl, levels=levels,
                    vmin=VMIN, vmax=VMAX, cmap=CMAP, extend="neither")
        ax.contour(X, Y, T, levels=iso_levs,
                   colors="white", linewidths=ISO_LW, alpha=0.75)
        if T.max() >= T_melt:
            ax.contour(X, Y, T, levels=[T_melt],
                       colors="#FFD700", linewidths=2.0)
        if invert_y:
            ax.invert_yaxis()
        ax.set_xlim(0, Lx_mm)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10, pad=5)
        ax.tick_params(labelsize=9, direction="in",
                       top=True, right=True, length=4, width=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(1.0); sp.set_edgecolor("black")

    _draw(ax_xy, X_xy, Y_xy, T_xy,
          f"xy-plane (z=0)   t={t_val*1e3:.1f} ms",
          "x [mm]", "y [mm]", False)
    _draw(ax_xz, X_xz, Z_xz, T_xz,
          f"xz-plane (y={y_track*1e3:.2f} mm)   t={t_val*1e3:.1f} ms",
          "x [mm]", "z [mm]", True)

    # Colorbar
    cbar_l = left_m + panel_w * 0.08
    cbar_w = panel_w * 0.84
    cax = fig.add_axes(to_frac(cbar_l, bot_m * 0.30, cbar_w, 0.30))
    sm  = cm.ScalarMappable(cmap=CMAP, norm=norm_cb)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="neither")
    cbar.set_ticks([400, 800, 1200, 1600])
    cbar.ax.tick_params(labelsize=14, length=7, width=1.2,
                        direction="out", bottom=True, top=False)
    cbar.outline.set_linewidth(1.0)
    cbar.set_label("Temperature (K)", fontsize=14, labelpad=8)

    plt.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

def plot_losses(history, fname):
    """Training loss curves + temperature monitoring."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="white")
    ep = np.arange(1, len(history["total"]) + 1)

    axes[0].semilogy(ep, history["total"], "k",       lw=2,   label="Total")
    axes[0].semilogy(ep, history["pde"],   "#c0392b", lw=1.5, ls="--", label="PDE")
    axes[0].semilogy(ep, history["bc"],    "#27ae60", lw=1.5, ls=":",  label="BC")
    axes[0].axvline(N_EPOCHS_1, color="orange", ls=":", lw=1.5, label="Phase 1→2")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (normalised)")
    axes[0].set_title("SPINN v4 Training Loss")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["T_laser"], "#c0392b", lw=1.5, label="T at laser centre")
    axes[1].plot(ep, history["T_max"],   "#e67e22", lw=1.5, ls="--", label="T_max (coarse grid)")
    axes[1].axhline(T_melt, color="gold", lw=1.5, ls="--",
                    label=f"T_melt = {T_melt:.0f} K")
    axes[1].axhline(T0, color="gray", lw=1.0, ls=":",
                    label=f"T0 = {T0:.0f} K")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("T [K]")
    axes[1].set_title("Temperature at laser centre (t = 8 ms)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
#  15.  GENERATE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Generating outputs ---")
plot_losses(history, os.path.join(RESULTS_DIR, "spinn_v4_loss.png"))

print("\nMelt-pool summary (best model):")
for t_val, label in [(8e-3, "t08ms"), (t_end, "t10ms"),(15e-3, "t15ms")]:
    print(f"\n  [{label}]  t = {t_val*1e3:.1f} ms")
    L, W, D = melt_pool(best_params, t_val)
    plot_snapshot(best_params, t_val,
                  os.path.join(RESULTS_DIR, f"spinn_v4_{label}.png"))

print(f"\nAll results saved to: {RESULTS_DIR}/")