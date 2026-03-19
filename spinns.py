"""
LPBF SPINN v3 — Hard IC enforcement, physical units, stable init
=================================================================
Key changes vs v2:
  1. Output ansatz: T = T0 + tau * NN(tau,xi,eta,zeta)
     → IC satisfied EXACTLY by construction (no IC loss term needed)
  2. Physical-unit PDE: work in K directly, divide residual by qv_peak
     → all residual terms O(1)
  3. Proper SPINN init: each axis MLP final layer output ~N(0, 0.01)
     → einsum product stays small at init
  4. Smaller rank (32) with proper scaling — avoids rank explosion
  5. Gradient clipping tightened to 0.5
  6. Monitoring prints T at laser centre exactly
"""

import os, time
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import jax
import jax.numpy as jnp
from jax import jvp, jit
import optax
import flax.linen as nn
from flax.training import train_state

jax.config.update("jax_enable_x64", False)

np.random.seed(42)
key = jax.random.PRNGKey(42)

RESULTS_DIR = "results_spinn_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
rho      = 8351.91
P_abs    = 150.0
r_beam   = 450.0e-6
c_depth  = 450.0e-6
v_scan   = 0.10
T0       = 300.0
T_inf    = 300.0
h_conv   = 10.0
T_melt   = 1533.0

Lx, Ly, Lz = 2.0e-3, 1.5e-3, 1.0e-3
x0_laser   = 0.50e-3
x1_laser   = 1.50e-3
y_track    = 0.50e-3
t_end      = (x1_laser - x0_laser) / v_scan   # 10 ms

# ─────────────────────────────────────────────────────────────────────────────
#  2.  MATERIAL PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────
def kappa_fn(T):
    return (229.87 + 0.0184 * T
            + 225.10 * jnp.tanh(0.018 * (T - 1816.8)))

def cp_fn(T):
    return (407.62 + 0.142 * T
            - 61.43  * jnp.exp(-3.1e-4 * (T - 798.0)**2)
            + 1054.96 * jnp.exp(-6.2e-5 * (T - 1816.8)**2))

kap0 = float(229.87 + 0.0184*T0 + 225.10*np.tanh(0.018*(T0-1816.8)))
cp0  = float(407.62 + 0.142*T0
             - 61.43*np.exp(-3.1e-4*(T0-798.0)**2)
             + 1054.96*np.exp(-6.2e-5*(T0-1816.8)**2))
alp0 = kap0 / (rho * cp0)

# ─────────────────────────────────────────────────────────────────────────────
#  3.  HEAT SOURCE
# ─────────────────────────────────────────────────────────────────────────────
qv_peak = (6.0 * np.sqrt(3.0) * P_abs /
           (np.pi * np.sqrt(np.pi) * r_beam**2 * c_depth))

def heat_source(x_p, y_p, z_p, t_p):
    x_laser = jnp.clip(x0_laser + v_scan * t_p,
                       a_min=float(x0_laser), a_max=float(x1_laser))
    return float(qv_peak) * (
        jnp.exp(-3.0 * ((x_p - x_laser)**2 + (y_p - float(y_track))**2)
                / float(r_beam)**2)
        * jnp.exp(-z_p**2 / float(c_depth)**2)
    )

# ─────────────────────────────────────────────────────────────────────────────
#  4.  NON-DIMENSIONALISATION
#
#  Spatial: ξ = x/r_beam,  η = y/r_beam,  ζ = z/c_depth
#  Time:    τ = t / t_ref,  t_ref = r_beam² / α0
#
#  Temperature: T [K] — kept in physical units
#
#  PDE residual normalised by qv_peak → all terms O(1)
# ─────────────────────────────────────────────────────────────────────────────
r_ref  = r_beam
t_ref  = r_beam**2 / alp0

x_nd_max = Lx / r_ref
y_nd_max = Ly / r_ref
z_nd_max = Lz / c_depth
t_nd_max = t_end / t_ref

x0_nd = x0_laser / r_ref
x1_nd = x1_laser / r_ref
yt_nd = y_track  / r_ref

# Characteristic temperature rise from analytical Rosenthal estimate
T_char = float(qv_peak) * float(t_ref) / (rho * cp0)   # ≈ 1000–3000 K

print("=" * 66)
print("  LPBF SPINN v3 — Hastelloy X")
print("=" * 66)
print(f"  r_beam  = {r_beam*1e6:.0f} µm  |  c_depth = {c_depth*1e6:.0f} µm")
print(f"  P_abs   = {P_abs:.1f} W  |  v_scan = {v_scan*1e3:.0f} mm/s")
print(f"  qv_peak = {qv_peak:.4e} W/m³")
print(f"  kap0    = {kap0:.2f} W/(m·K)  |  cp0 = {cp0:.1f} J/(kg·K)")
print(f"  alp0    = {alp0:.4e} m²/s")
print(f"  t_ref   = {t_ref*1e3:.4f} ms")
print(f"  T_char  = {T_char:.1f} K  (Rosenthal estimate)")
print(f"  t_nd_max = {t_nd_max:.4f}")
print(f"  Domain nd: ξ∈[0,{x_nd_max:.2f}]  η∈[0,{y_nd_max:.2f}]  ζ∈[0,{z_nd_max:.2f}]")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  5.  SPINN ARCHITECTURE
#
#  Hard IC ansatz:  T(τ,ξ,η,ζ) = T0 + τ · NN(τ,ξ,η,ζ)
#
#  NN is the raw SPINN output (can be any sign — no softplus needed).
#  At τ=0: T = T0 exactly, regardless of NN.
#
#  Init target: NN ≈ 0 everywhere at init → T ≈ T0 everywhere.
#  Achieved by: small kernel_init (std=0.01) on the LAST Dense layer
#               of each axis MLP, zero bias.
#  Then einsum product of 4 small vectors ≈ 0.  ✓
#
#  Rank: 32 (sufficient for 4D LPBF, avoids memory explosion)
#  Depth: 4 hidden layers × 64 units
# ─────────────────────────────────────────────────────────────────────────────
RANK     = 32
N_HIDDEN = 64
N_LAYERS = 4

SMALL_INIT = nn.initializers.normal(stddev=0.01)

class AxisMLP(nn.Module):
    hidden: Sequence[int]
    rank: int

    @nn.compact
    def __call__(self, x):
        # x: (N,) → h: (N, 1)
        h = x[:, None]
        for feat in self.hidden:
            h = nn.Dense(feat,
                         kernel_init=nn.initializers.glorot_normal(),
                         bias_init=nn.initializers.zeros)(h)
            h = nn.tanh(h)
        # Final layer: tiny init → output ≈ 0 at init
        # With rank=32 and 4 axis nets: einsum ≈ 32 × (0.01)^4 ≈ 3e-9 ≈ 0 ✓
        return nn.Dense(self.rank,
                        kernel_init=SMALL_INIT,
                        bias_init=nn.initializers.zeros)(h)

class SPINN_v3(nn.Module):
    hidden: Sequence[int]
    rank: int

    def setup(self):
        kw = dict(hidden=self.hidden, rank=self.rank)
        self.net_t = AxisMLP(**kw)
        self.net_x = AxisMLP(**kw)
        self.net_y = AxisMLP(**kw)
        self.net_z = AxisMLP(**kw)

    def __call__(self, tau, xi, eta, zeta):
        """
        Returns raw NN output (no activation), shape (Nt, Nx, Ny, Nz).
        Physical T = T0 + tau[:,None,None,None] * output
        """
        Ft = self.net_t(tau)    # (Nt, rank)
        Fx = self.net_x(xi)     # (Nx, rank)
        Fy = self.net_y(eta)    # (Ny, rank)
        Fz = self.net_z(zeta)   # (Nz, rank)
        # CP outer product
        return jnp.einsum('tr,xr,yr,zr->txyz', Ft, Fx, Fy, Fz)

# ─────────────────────────────────────────────────────────────────────────────
#  6.  PHYSICAL TEMPERATURE (with hard IC)
# ─────────────────────────────────────────────────────────────────────────────
model    = SPINN_v3(hidden=tuple([N_HIDDEN]*N_LAYERS), rank=RANK)
apply_fn = model.apply

key, sk = jax.random.split(key)
dummy_t = jnp.linspace(0.0, t_nd_max, 4)
dummy_x = jnp.linspace(0.0, x_nd_max, 4)
dummy_y = jnp.linspace(0.0, y_nd_max, 4)
dummy_z = jnp.linspace(0.0, z_nd_max, 4)
params  = model.init(sk, dummy_t, dummy_x, dummy_y, dummy_z)

n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

def T_phys(params, tau, xi, eta, zeta):
    """Physical temperature [K], shape (Nt, Nx, Ny, Nz). Hard IC: T(0)=T0."""
    raw = apply_fn(params, tau, xi, eta, zeta)
    # tau broadcast: (Nt,) → (Nt,1,1,1)
    tau4 = tau[:, None, None, None]
    return float(T0) + tau4 * raw * float(T_char)

# Check init
T_init = np.array(T_phys(params, dummy_t, dummy_x, dummy_y, dummy_z))
print(f"  SPINN parameters : {n_params:,}")
print(f"  raw NN at init   : [{float(apply_fn(params,dummy_t,dummy_x,dummy_y,dummy_z).min()):.4e}, "
      f"{float(apply_fn(params,dummy_t,dummy_x,dummy_y,dummy_z).max()):.4e}]")
print(f"  T at init        : [{T_init.min():.1f}, {T_init.max():.1f}] K")
print(f"  (target: all ≈ {T0:.0f} K at τ=0, rising for τ>0)")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  7.  LOSS FUNCTIONS
#
#  PDE (physical units, normalised by qv_peak):
#
#    R = ρ·cp(T)·(∂T/∂t) - ∇·(k(T)·∇T) - q_v = 0
#
#  In non-dim coords (ξ,η,ζ,τ):
#    ∂T/∂t = (1/t_ref)·∂T/∂τ
#    ∂T/∂x = (1/r_ref)·∂T/∂ξ,  ∂T/∂z = (1/c_depth)·∂T/∂ζ
#
#  Normalise residual by qv_peak → R_nd = R / qv_peak ≈ O(1)
#
#  BC: Robin on top (z=0), Neumann elsewhere.
#  IC: Satisfied exactly by ansatz — NO IC LOSS TERM.
# ─────────────────────────────────────────────────────────────────────────────
_r2 = float(r_ref)**2
_c2 = float(c_depth)**2
_tr = float(t_ref)
_rr = float(r_ref)
_cd = float(c_depth)
_qp = float(qv_peak)
_rho = float(rho)

def pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c):
    ones = lambda v: jnp.ones_like(v)

    # ── ∂T/∂τ ────────────────────────────────────────────────────────────
    T, dT_dtau = jvp(
        lambda t: T_phys(params, t, xi_c, eta_c, zeta_c),
        (tau_c,), (ones(tau_c),))
    dT_dt = dT_dtau / _tr   # physical time derivative

    T_K = T   # shape (Nt, Nx, Ny, Nz)
    k_T  = kappa_fn(T_K)
    cp_T = cp_fn(T_K)

    # dk/dT
    dkdT = (0.0184 + 225.10 * 0.018
            * (1.0 - jnp.tanh(0.018 * (T_K - 1816.8))**2))

    # ── ∂T/∂ξ, ∂²T/∂ξ² ──────────────────────────────────────────────────
    _, dT_dxi = jvp(
        lambda x: T_phys(params, tau_c, x, eta_c, zeta_c),
        (xi_c,), (ones(xi_c),))
    _, d2T_dxi2 = jvp(
        lambda x: jvp(
            lambda x2: T_phys(params, tau_c, x2, eta_c, zeta_c),
            (x,), (ones(x),))[1],
        (xi_c,), (ones(xi_c),))

    # ── ∂T/∂η, ∂²T/∂η² ──────────────────────────────────────────────────
    _, dT_deta = jvp(
        lambda y: T_phys(params, tau_c, xi_c, y, zeta_c),
        (eta_c,), (ones(eta_c),))
    _, d2T_deta2 = jvp(
        lambda y: jvp(
            lambda y2: T_phys(params, tau_c, xi_c, y2, zeta_c),
            (y,), (ones(y),))[1],
        (eta_c,), (ones(eta_c),))

    # ── ∂T/∂ζ, ∂²T/∂ζ² ──────────────────────────────────────────────────
    _, dT_dzeta = jvp(
        lambda z: T_phys(params, tau_c, xi_c, eta_c, z),
        (zeta_c,), (ones(zeta_c),))
    _, d2T_dzeta2 = jvp(
        lambda z: jvp(
            lambda z2: T_phys(params, tau_c, xi_c, eta_c, z2),
            (z,), (ones(z),))[1],
        (zeta_c,), (ones(zeta_c),))

    # Physical 2nd derivatives (chain rule for coord transform)
    d2T_dx2 = d2T_dxi2   / _r2
    d2T_dy2 = d2T_deta2  / _r2
    d2T_dz2 = d2T_dzeta2 / _c2

    dT_dx = dT_dxi  / _rr
    dT_dy = dT_deta / _rr
    dT_dz = dT_dzeta / _cd

    # Heat source at collocation points
    tau4  = tau_c[:, None, None, None]
    xi4   = xi_c[None, :, None, None]
    eta4  = eta_c[None, None, :, None]
    zeta4 = zeta_c[None, None, None, :]
    q_v   = heat_source(xi4*_rr, eta4*_rr, zeta4*_cd, tau4*_tr)

    # PDE residual (physical): ρ·cp·∂T/∂t - ∇·(k∇T) - q_v
    # ∇·(k∇T) = k·∇²T + ∇k·∇T = k·(d2x+d2y+d2z) + dkdT·(|∇T|²)
    laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
    grad_sq   = dT_dx**2 + dT_dy**2 + dT_dz**2

    R = (_rho * cp_T * dT_dt
         - k_T * laplacian
         - dkdT * grad_sq
         - q_v)

    # Normalise by qv_peak → O(1)
    R_nd = R / _qp
    return jnp.mean(R_nd**2)

def bc_loss_fn(params, tau_b, xi_b, eta_b, zeta_b):
    """
    Top z=0 (ζ=0): Robin  -k·∂T/∂z = h_conv·(T - T_inf)
    Other 5 faces: Neumann ∂T/∂n = 0
    Normalise by h_conv·T_char to get O(1).
    """
    ones = lambda v: jnp.ones_like(v)
    loss = jnp.array(0.0)
    norm = float(h_conv) * float(T_char)

    # Top ζ=0: Robin
    zeta_top = jnp.zeros(1)
    T_top, dT_dzeta_top = jvp(
        lambda z: T_phys(params, tau_b, xi_b, eta_b, z),
        (zeta_top,), (ones(zeta_top),))
    dT_dz_top = dT_dzeta_top / _cd
    k_top = kappa_fn(T_top)
    res_robin = (-k_top * dT_dz_top - float(h_conv) * (T_top - float(T_inf))) / norm
    loss = loss + jnp.mean(res_robin**2)

    # Bottom ζ=z_nd_max: Neumann
    zeta_bot = jnp.array([z_nd_max])
    _, dT_dz_bot = jvp(
        lambda z: T_phys(params, tau_b, xi_b, eta_b, z),
        (zeta_bot,), (ones(zeta_bot),))
    loss = loss + jnp.mean((dT_dz_bot / _cd)**2) / norm**2

    # Left ξ=0: Neumann
    xi_left = jnp.zeros(1)
    _, dT_dx_left = jvp(
        lambda x: T_phys(params, tau_b, x, eta_b, zeta_b),
        (xi_left,), (ones(xi_left),))
    loss = loss + jnp.mean((dT_dx_left / _rr)**2) / norm**2

    # Right ξ=x_nd_max: Neumann
    xi_right = jnp.array([x_nd_max])
    _, dT_dx_right = jvp(
        lambda x: T_phys(params, tau_b, x, eta_b, zeta_b),
        (xi_right,), (ones(xi_right),))
    loss = loss + jnp.mean((dT_dx_right / _rr)**2) / norm**2

    # Front η=0: Neumann
    eta_front = jnp.zeros(1)
    _, dT_dy_front = jvp(
        lambda y: T_phys(params, tau_b, xi_b, y, zeta_b),
        (eta_front,), (ones(eta_front),))
    loss = loss + jnp.mean((dT_dy_front / _rr)**2) / norm**2

    # Back η=y_nd_max: Neumann
    eta_back = jnp.array([y_nd_max])
    _, dT_dy_back = jvp(
        lambda y: T_phys(params, tau_b, xi_b, y, zeta_b),
        (eta_back,), (ones(eta_back),))
    loss = loss + jnp.mean((dT_dy_back / _rr)**2) / norm**2

    return loss / 6.0

def total_loss_fn(params,
                  tau_c, xi_c, eta_c, zeta_c,
                  tau_b, xi_b, eta_b, zeta_b,
                  w_pde, w_bc):
    Lp = pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c)
    Lb = bc_loss_fn(params, tau_b, xi_b, eta_b, zeta_b)
    return w_pde * Lp + w_bc * Lb, (Lp, Lb)

# ─────────────────────────────────────────────────────────────────────────────
#  8.  COLLOCATION SAMPLING
#
#  SPINN uses factorizable (per-axis) coordinates.
#  We sample each axis independently, then the full grid is their product.
#
#  Laser-focused: 60% of ξ points near laser, 60% of η near track centre.
#  This ensures the laser hotspot is well-sampled.
# ─────────────────────────────────────────────────────────────────────────────
N_T, N_X, N_Y, N_Z = 128, 128, 128, 128   # per-axis → 16×18×14×12 = 48,384 pts
N_BC = 16

LASER_ND_STD = 1.2   # ≈ 1.2 × r_beam near laser in x and y

def sample_col(key):
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    # τ: uniform
    tau_c = jax.random.uniform(k1, (N_T,), minval=0.0, maxval=t_nd_max)

    # ξ: 40% uniform + 60% near laser
    n_xu = max(2, N_X * 4 // 10)
    n_xf = N_X - n_xu
    xi_u = jax.random.uniform(k2, (n_xu,), minval=0.0, maxval=x_nd_max)
    # Laser position spans x0_nd to x1_nd — sample uniformly along track
    xi_laser = jax.random.uniform(k3, (n_xf,), minval=x0_nd, maxval=x1_nd)
    xi_f = jnp.clip(xi_laser + jax.random.normal(k4, (n_xf,)) * LASER_ND_STD,
                    0.0, x_nd_max)
    xi_c = jnp.concatenate([xi_u, xi_f])

    # η: 40% uniform + 60% near track centreline
    n_yu = max(2, N_Y * 4 // 10)
    n_yf = N_Y - n_yu
    eta_u = jax.random.uniform(k5, (n_yu,), minval=0.0, maxval=y_nd_max)
    eta_f = jnp.clip(float(yt_nd) + jax.random.normal(k6, (n_yf,)) * LASER_ND_STD,
                     0.0, y_nd_max)
    eta_c = jnp.concatenate([eta_u, eta_f])

    # ζ: 40% uniform + 60% near surface (exponential)
    n_zu = max(2, N_Z * 4 // 10)
    n_zf = N_Z - n_zu
    zeta_u = jax.random.uniform(k7, (n_zu,), minval=0.0, maxval=z_nd_max)
    zeta_f = jnp.clip(jax.random.exponential(k8, (n_zf,)) * 0.6,
                      0.0, z_nd_max)
    zeta_c = jnp.concatenate([zeta_u, zeta_f])

    return tau_c, xi_c, eta_c, zeta_c

def sample_bc(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tau_b  = jax.random.uniform(k1, (N_BC,), minval=0.0, maxval=t_nd_max)
    xi_b   = jax.random.uniform(k2, (N_BC,), minval=0.0, maxval=x_nd_max)
    eta_b  = jax.random.uniform(k3, (N_BC,), minval=0.0, maxval=y_nd_max)
    zeta_b = jax.random.uniform(k4, (N_BC,), minval=0.0, maxval=z_nd_max)
    return tau_b, xi_b, eta_b, zeta_b

# ─────────────────────────────────────────────────────────────────────────────
#  9.  TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
#  10.  OPTIMISER — Two-phase with warm-up
# ─────────────────────────────────────────────────────────────────────────────
N_EPOCHS_1 = 12_000   # Phase 1: cosine decay
N_EPOCHS_2 =  8_000   # Phase 2: fine-tune
N_EPOCHS   = N_EPOCHS_1 + N_EPOCHS_2

LR_PEAK = 5e-4
LR_MIN  = 1e-5

# Warm-up for 500 steps then cosine decay
warmup   = optax.linear_schedule(0.0, LR_PEAK, 500)
cosine   = optax.cosine_decay_schedule(LR_PEAK, N_EPOCHS_1 - 500, LR_MIN / LR_PEAK)
lr_sched = optax.join_schedules([warmup, cosine], [500])

def make_opt(lr):
    return optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(lr)
    )

state = train_state.TrainState.create(
    apply_fn=apply_fn, params=params,
    tx=make_opt(lr_sched))

w_pde = jnp.array(1.0)
w_bc  = jnp.array(0.05)   # BC is secondary — IC is exact by construction

print(f"  Interior pts/step : {N_T*N_X*N_Y*N_Z:,}  ({N_T}×{N_X}×{N_Y}×{N_Z})")
print(f"  Phase 1 epochs    : {N_EPOCHS_1:,}  (warm-up 500 + cosine {LR_PEAK:.0e}→{LR_MIN:.0e})")
print(f"  Phase 2 epochs    : {N_EPOCHS_2:,}  (Adam fine-tune {LR_MIN:.0e})")
print(f"  Loss weights      : w_pde={float(w_pde):.2f}  w_bc={float(w_bc):.2f}")
print(f"  IC: satisfied EXACTLY by construction (no IC loss term)")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  11.  MONITORING — Evaluate T at laser centre
# ─────────────────────────────────────────────────────────────────────────────
# At t=8ms: laser at x = x0 + v*t = 0.5 + 0.1*0.008 = 1.3 mm → ξ = 2.89
tau_8ms  = jnp.array([8e-3 / t_ref])
xi_8ms   = jnp.array([1.3e-3 / r_ref])   # laser centre x
eta_8ms  = jnp.array([y_track / r_ref])   # laser centre y
zeta_8ms = jnp.array([0.0])               # surface z=0

# Wider grid for T_max search
xi_mon   = jnp.linspace(0.0, x_nd_max, 20)
eta_mon  = jnp.linspace(0.0, y_nd_max, 14)
zeta_mon = jnp.linspace(0.0, 2.0, 8)

print(f"  Monitoring: T at laser centre (t=8ms, x=1.3mm, y=0.5mm, z=0)")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  12.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
LOG_EVERY   = 500
best_loss   = np.inf
best_params = params
history     = {"total": [], "pde": [], "bc": [],
               "T_laser": [], "T_max": []}

print("=" * 66)
print("  SPINN v3 Training")
print("=" * 66)
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):

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

    # T at laser centre
    T_laser = float(T_phys(state.params,
                            tau_8ms, xi_8ms, eta_8ms, zeta_8ms)[0, 0, 0, 0])
    # T_max over wider grid
    T_grid  = np.array(T_phys(state.params, tau_8ms, xi_mon, eta_mon, zeta_mon))
    T_max   = float(T_grid.max())

    history["total"].append(lv)
    history["pde"].append(pv)
    history["bc"].append(bv)
    history["T_laser"].append(T_laser)
    history["T_max"].append(T_max)

    if lv < best_loss:
        best_loss   = lv
        best_params = state.params

    if epoch % LOG_EVERY == 0:
        if epoch <= N_EPOCHS_1:
            lr_now = float(lr_sched(epoch))
        else:
            lr_now = LR_MIN
        print(f"  Ep {epoch:6d}/{N_EPOCHS} | "
              f"L={lv:.3e}  PDE={pv:.3e}  BC={bv:.3e} | "
              f"LR={lr_now:.2e} | "
              f"T_laser={T_laser:.0f}K  T_max={T_max:.0f}K | "
              f"t={time.time()-t0:.0f}s")

print(f"\n  Done in {time.time()-t0:.1f}s  |  best loss = {best_loss:.4e}")

# ─────────────────────────────────────────────────────────────────────────────
#  13.  EVALUATION & PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
CMAP, VMIN, VMAX = "RdBu_r", 300.0, 2000.0
Lx_mm, Ly_mm, Lz_mm = Lx*1e3, Ly*1e3, Lz*1e3

def eval_xy(params, t_val, res=150):
    tau  = jnp.array([t_val / t_ref])
    xi   = jnp.linspace(0.0, x_nd_max, res)
    eta  = jnp.linspace(0.0, y_nd_max, res)
    zeta = jnp.zeros(1)
    T    = np.array(T_phys(params, tau, xi, eta, zeta))[0, :, :, 0]
    X    = np.linspace(0, Lx_mm, res)
    Y    = np.linspace(0, Ly_mm, res)
    return np.meshgrid(X, Y, indexing='ij')[0].T, \
           np.meshgrid(X, Y, indexing='ij')[1].T, T.T

def eval_xz(params, t_val, res=150):
    tau  = jnp.array([t_val / t_ref])
    xi   = jnp.linspace(0.0, x_nd_max, res)
    eta  = jnp.array([y_track / r_ref])
    zeta = jnp.linspace(0.0, z_nd_max, res)
    T    = np.array(T_phys(params, tau, xi, eta, zeta))[0, :, 0, :]
    X    = np.linspace(0, Lx_mm, res)
    Z    = np.linspace(0, Lz_mm, res)
    return np.meshgrid(X, Z, indexing='ij')[0].T, \
           np.meshgrid(X, Z, indexing='ij')[1].T, T.T

def melt_pool(params, t_val, res=80):
    _, _, T_xy = eval_xy(params, t_val, res)
    _, _, T_xz = eval_xz(params, t_val, res)
    print(f"  T_surf ∈ [{T_xy.min():.0f}, {T_xy.max():.0f}] K")
    print(f"  T_xz   ∈ [{T_xz.min():.0f}, {T_xz.max():.0f}] K")
    if T_xy.max() < T_melt:
        print("  No melt pool detected.")
        return 0.0, 0.0, 0.0
    X_mm = np.linspace(0, Lx_mm, res)
    Y_mm = np.linspace(0, Ly_mm, res)
    Z_mm = np.linspace(0, Lz_mm, res)
    r, c = np.where(T_xy >= T_melt)
    L = float(X_mm[r.max()] - X_mm[r.min()])
    W = float(Y_mm[c.max()] - Y_mm[c.min()])
    rz, _ = np.where(T_xz >= T_melt) if T_xz.max() >= T_melt else (np.array([0]), None)
    D = float(Z_mm[rz.max()] - Z_mm[rz.min()]) if T_xz.max() >= T_melt else 0.0
    print(f"  Melt pool: L={L:.3f} mm  W={W:.3f} mm  D={D:.3f} mm")
    return L, W, D

def plot_snapshot(params, t_val, fname, res=150):
    X_xy, Y_xy, T_xy = eval_xy(params, t_val, res)
    X_xz, Z_xz, T_xz = eval_xz(params, t_val, res)
    vmax = max(float(T_xy.max()), float(T_xz.max()), VMAX)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), facecolor="white")
    for ax, X, Y, T, title, yl, inv in [
        (axes[0], X_xy, Y_xy, T_xy,
         f"xy-plane (z=0)   t={t_val*1e3:.1f} ms", "y [mm]", False),
        (axes[1], X_xz, Z_xz, T_xz,
         f"xz-plane (y={y_track*1e3:.2f} mm)   t={t_val*1e3:.1f} ms", "z [mm]", True),
    ]:
        levels = np.linspace(VMIN, vmax, 41)
        cf = ax.contourf(X, Y, np.clip(T, VMIN, vmax),
                         levels=levels, cmap=CMAP, extend="neither")
        ax.contour(X, Y, T, levels=np.linspace(VMIN, vmax, 17),
                   colors="white", linewidths=0.5, alpha=0.7)
        if T.max() >= T_melt:
            ax.contour(X, Y, T, levels=[T_melt],
                       colors="yellow", linewidths=1.5)
        if inv:
            ax.invert_yaxis()
        ax.set_xlabel("x [mm]", fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontsize=10)
        plt.colorbar(cf, ax=ax, label="T [K]")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

def plot_losses(history, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor="white")
    ep = np.arange(1, len(history["total"]) + 1)

    axes[0].semilogy(ep, history["total"], "k",      lw=2,   label="Total")
    axes[0].semilogy(ep, history["pde"],   "#c0392b", lw=1.5, ls="--", label="PDE")
    axes[0].semilogy(ep, history["bc"],    "#27ae60", lw=1.5, ls=":",  label="BC")
    axes[0].axvline(N_EPOCHS_1, color="orange", ls=":", lw=1.5, label="Phase 1→2")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("SPINN v3 Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["T_laser"], "#c0392b", lw=1.5, label="T at laser centre")
    axes[1].plot(ep, history["T_max"],   "#e67e22", lw=1.5, ls="--", label="T_max (grid)")
    axes[1].axhline(T_melt, color="gold",  lw=1.5, ls="--", label=f"T_melt={T_melt:.0f}K")
    axes[1].axhline(T0,     color="gray",  lw=1.0, ls=":",  label=f"T0={T0:.0f}K")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("T [K]")
    axes[1].set_title("T at laser centre (t=8ms)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  14.  OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generating outputs ---")
plot_losses(history, os.path.join(RESULTS_DIR, "spinn_v3_loss.png"))

print("\nMelt-pool summary (best model):")
print(f"  {'t [ms]':>8}  {'L [mm]':>8}  {'W [mm]':>8}  {'D [mm]':>8}")

for t_val, label in [(8e-3, "t08ms"), (t_end, "t10ms")]:
    print(f"\n  [{label}]  t = {t_val*1e3:.1f} ms")
    L, W, D = melt_pool(best_params, t_val)
    plot_snapshot(best_params, t_val,
                  os.path.join(RESULTS_DIR, f"spinn_v3_{label}.png"))
    print(f"  {t_val*1e3:8.1f}  {L:8.3f}  {W:8.3f}  {D:8.3f}")

print(f"\nAll results in: {RESULTS_DIR}/")