"""
LPBF SPINN v4 — Hastelloy X, Single Track
==========================================
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
import pickle

import jax
import jax.numpy as jnp
from jax import jvp, jit
import optax
import flax.linen as nn
from flax.training import train_state

jax.config.update("jax_enable_x64", True)

np.random.seed(42)
key = jax.random.PRNGKey(42)

RESULTS_DIR = "results_spinn_v4"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  1.  PHYSICAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
rho      = 8351.91
P_abs    = 150.0
r_beam   = 450.0e-6
c_depth  = 500.0e-6
v_scan   = 0.10
T0       = 300.0
T_inf    = 300.0
h_conv   = 10.0
T_melt   = 1533.0

Lx, Ly, Lz = 2.0e-3, 1.5e-3, 1.0e-3

x0_laser = 0.50e-3
x1_laser = 1.50e-3
y_track  = 0.50e-3

t_end = (x1_laser - x0_laser) / v_scan

# ══════════════════════════════════════════════════════════════════════════════
#  2.  MATERIAL PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
def kappa_fn(T):
    return (229.87 + 0.0184 * T
            + 225.10 * jnp.tanh(0.0118 * (T - 1816.8)))

def cp_fn(T):
    return (407.62 + 0.142 * T
            - 61.43  * jnp.exp(-3.1e-4 * (T - 798.0)**2)
            + 1054.96 * jnp.exp(-6.2e-5 * (T - 1816.8)**2))

def dkappa_dT_fn(T):
    return 0.0184 + 225.10 * 0.0118 * (1.0 - jnp.tanh(0.0118*(T - 1816.8))**2)

kap0 = float(229.87 + 0.0184*T0 + 225.10*np.tanh(0.0118*(T0-1816.8)))
cp0  = float(407.62 + 0.142*T0
             - 61.43*np.exp(-3.1e-4*(T0-798.0)**2)
             + 1054.96*np.exp(-6.2e-5*(T0-1816.8)**2))
alp0 = kap0 / (rho * cp0)

# ══════════════════════════════════════════════════════════════════════════════
#  3.  NON-DIMENSIONALISATION
# ══════════════════════════════════════════════════════════════════════════════
r_ref  = r_beam
t_ref  = r_beam**2 / alp0

x_nd_max = Lx      / r_ref
y_nd_max = Ly      / r_ref
z_nd_max = Lz      / c_depth
t_nd_max = t_end   / t_ref

x0_nd = x0_laser / r_ref
x1_nd = x1_laser / r_ref
yt_nd = y_track  / r_ref

qv_peak = (6.0 * np.sqrt(3.0) * P_abs /
           (np.pi * np.sqrt(np.pi) * r_beam**2 * c_depth))

T_char = float(qv_peak) * float(t_ref) / (rho * cp0)

_pde_scale = rho * cp0 * T_char / t_ref

print("=" * 60)
print("  LPBF SPINN v4 — Hastelloy X, Single Track")
print("=" * 60)
print(f"  Domain  : {Lx*1e3:.1f} x {Ly*1e3:.1f} x {Lz*1e3:.1f} mm")
print(f"  P_abs   : {P_abs:.1f} W  |  v_scan: {v_scan*1e3:.0f} mm/s")
print(f"  T0={T0:.0f} K  |  T_melt={T_melt:.0f} K")
print(f"  T_char  = {T_char:.1f} K  |  t_ref = {t_ref*1e3:.4f} ms")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  4.  HEAT SOURCE
# ══════════════════════════════════════════════════════════════════════════════
def heat_source(x_p, y_p, z_p, t_p):
    x_laser = jnp.clip(x0_laser + v_scan * t_p,
                       min=float(x0_laser),
                       max=float(x1_laser))
    return float(qv_peak) * (
        jnp.exp(-3.0 * ((x_p - x_laser)**2 + (y_p - float(y_track))**2)
                / float(r_beam)**2)
        * jnp.exp(-3.0 * (z_p**2 / float(c_depth)**2))
    )

# ══════════════════════════════════════════════════════════════════════════════
#  5.  SPINN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
RANK     = 64
N_HIDDEN = 64
N_LAYERS = 5

OUTPUT_INIT = nn.initializers.normal(stddev=0.1)

class AxisMLP(nn.Module):
    hidden: Sequence[int]
    rank: int

    @nn.compact
    def __call__(self, x):
        h = x[:, None]
        for feat in self.hidden:
            h = nn.Dense(feat,
                         kernel_init=nn.initializers.glorot_normal(),
                         bias_init=nn.initializers.zeros)(h)
            h = nn.tanh(h)
        return nn.Dense(self.rank,
                        kernel_init=OUTPUT_INIT,
                        bias_init=nn.initializers.zeros)(h)

class SPINN_v4(nn.Module):
    hidden: Sequence[int]
    rank: int

    def setup(self):
        kw = dict(hidden=self.hidden, rank=self.rank)
        self.net_t = AxisMLP(**kw)
        self.net_x = AxisMLP(**kw)
        self.net_y = AxisMLP(**kw)
        self.net_z = AxisMLP(**kw)

    def __call__(self, tau, xi, eta, zeta):
        Ft = self.net_t(tau)
        Fx = self.net_x(xi)
        Fy = self.net_y(eta)
        Fz = self.net_z(zeta)
        return jnp.einsum('tr,xr,yr,zr->txyz', Ft, Fx, Fy, Fz)

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
# ══════════════════════════════════════════════════════════════════════════════
def T_phys(params, tau, xi, eta, zeta):
    raw  = apply_fn(params, tau, xi, eta, zeta)
    tau4 = tau[:, None, None, None]
    return float(T0) + tau4 * raw * float(T_char)

T_init = np.array(T_phys(params, dummy_t, dummy_x, dummy_y, dummy_z))


# ══════════════════════════════════════════════════════════════════════════════
#  7.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
_r2  = float(r_ref)**2
_c2  = float(c_depth)**2
_tr  = float(t_ref)
_rr  = float(r_ref)
_cd  = float(c_depth)
_qp  = float(qv_peak)
_rho = float(rho)
_ps  = float(_pde_scale)

def pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c):
    ones = lambda v: jnp.ones_like(v)

    T, dT_dtau = jvp(
        lambda t: T_phys(params, t, xi_c, eta_c, zeta_c),
        (tau_c,), (ones(tau_c),))
    dT_dt = dT_dtau / _tr

    T_K   = T
    k_T   = kappa_fn(T_K)
    cp_T  = cp_fn(T_K)
    dkdT  = dkappa_dT_fn(T_K)

    _, dT_dxi = jvp(
        lambda x: T_phys(params, tau_c, x, eta_c, zeta_c),
        (xi_c,), (ones(xi_c),))
    _, dT_deta = jvp(
        lambda y: T_phys(params, tau_c, xi_c, y, zeta_c),
        (eta_c,), (ones(eta_c),))
    _, dT_dzeta = jvp(
        lambda z: T_phys(params, tau_c, xi_c, eta_c, z),
        (zeta_c,), (ones(zeta_c),))

    dT_dx = dT_dxi   / _rr
    dT_dy = dT_deta  / _rr
    dT_dz = dT_dzeta / _cd

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

    d2T_dx2 = d2T_dxi2   / _r2
    d2T_dy2 = d2T_deta2  / _r2
    d2T_dz2 = d2T_dzeta2 / _c2

    tau4  = tau_c[:, None, None, None]
    xi4   = xi_c [None, :, None, None]
    eta4  = eta_c[None, None, :, None]
    zeta4 = zeta_c[None, None, None, :]
    q_v   = heat_source(xi4*_rr, eta4*_rr, zeta4*_cd, tau4*_tr)

    laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
    grad_sq   = dT_dx**2 + dT_dy**2 + dT_dz**2

    R = (_rho * cp_T * dT_dt
         - k_T * laplacian
         - dkdT * grad_sq
         - q_v)

    R_nd = R / _ps
    return jnp.mean(R_nd**2)

def bc_loss_fn(params, tau_b, xi_b, eta_b, zeta_b):

    ones = lambda v: jnp.ones_like(v)

    bc_scale = _ps * _rr

    loss = jnp.array(0.0)

    # ── Top surface z=0: Robin BC ─────────────────────────────────────────
    zeta_top = jnp.zeros(1)
    T_top, dT_dzeta_top = jvp(
        lambda z: T_phys(params, tau_b, xi_b, eta_b, z),
        (zeta_top,), (ones(zeta_top),))
    dT_dz_top = dT_dzeta_top / _cd
    k_top     = kappa_fn(T_top)
    res_robin = (-k_top * dT_dz_top
                 - float(h_conv) * (T_top - float(T_inf)))
    loss = loss + jnp.mean((res_robin / bc_scale)**2)


    zeta_bot = jnp.array([z_nd_max])
    T_bot = T_phys(params, tau_b, xi_b, eta_b, zeta_bot)
    res_dir = (T_bot - float(T0)) / float(T_char)
    loss = loss + jnp.mean(res_dir**2)

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

    return loss / 6.0

def total_loss_fn(params,
                  tau_c, xi_c, eta_c, zeta_c,
                  tau_b, xi_b, eta_b, zeta_b,
                  w_pde, w_bc):
 
    Lp = pde_loss_fn(params, tau_c, xi_c, eta_c, zeta_c)
    Lb = bc_loss_fn(params,  tau_b, xi_b, eta_b, zeta_b)
    return w_pde * Lp + w_bc * Lb, (Lp, Lb)

# ══════════════════════════════════════════════════════════════════════════════
#  8.  COLLOCATION POINT SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
N_T, N_X, N_Y, N_Z = 64, 128, 96, 48
N_BC = 96
LASER_ND_STD = 1

def sample_col(key):
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    tau_c = jax.random.uniform(k1, (N_T,), minval=0.0, maxval=t_nd_max)

    n_xu = max(2, N_X * 4 // 10)
    n_xf = N_X - n_xu
    xi_u = jax.random.uniform(k2, (n_xu,), minval=0.0, maxval=x_nd_max)
    xi_c_base = jax.random.uniform(k3, (n_xf,), minval=x0_nd, maxval=x1_nd)
    xi_f = jnp.clip(xi_c_base + jax.random.normal(k4, (n_xf,)) * LASER_ND_STD,
                    0.0, x_nd_max)
    xi_c = jnp.concatenate([xi_u, xi_f])

    n_yu = max(2, N_Y * 4 // 10)
    n_yf = N_Y - n_yu
    eta_u = jax.random.uniform(k5, (n_yu,), minval=0.0, maxval=y_nd_max)
    eta_f = jnp.clip(float(yt_nd) + jax.random.normal(k6, (n_yf,)) * LASER_ND_STD,
                     0.0, y_nd_max)
    eta_c = jnp.concatenate([eta_u, eta_f])

    n_zu = max(2, N_Z * 4 // 10)
    n_zf = N_Z - n_zu
    zeta_u = jax.random.uniform(k7, (n_zu,), minval=0.0, maxval=z_nd_max)
    zeta_f = jnp.clip(jax.random.exponential(k8, (n_zf,)) * 0.6, 0.0, z_nd_max)
    zeta_c = jnp.concatenate([zeta_u, zeta_f])

    return tau_c, xi_c, eta_c, zeta_c

def sample_bc(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tau_b  = jax.random.uniform(k1, (N_BC,), minval=0.0, maxval=t_nd_max)
    xi_b   = jax.random.uniform(k2, (N_BC,), minval=0.0, maxval=x_nd_max)
    eta_b  = jax.random.uniform(k3, (N_BC,), minval=0.0, maxval=y_nd_max)
    zeta_b = jax.random.uniform(k4, (N_BC,), minval=0.0, maxval=z_nd_max)
    return tau_b, xi_b, eta_b, zeta_b

# ══════════════════════════════════════════════════════════════════════════════
#  9.  TRAINING STEP
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
#  10.  OPTIMISER
# ══════════════════════════════════════════════════════════════════════════════
N_EPOCHS_1 = 15000
N_EPOCHS_2 = 15000
N_EPOCHS   = N_EPOCHS_1 + N_EPOCHS_2

LR_PEAK = 8e-4
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

w_pde = jnp.array(1.0)
w_bc  = jnp.array(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  11.  MONITORING POINTS
# ══════════════════════════════════════════════════════════════════════════════
tau_8ms  = jnp.array([8e-3 / t_ref])
xi_8ms   = jnp.array([(x0_laser + v_scan * 8e-3) / r_ref])
eta_8ms  = jnp.array([y_track / r_ref])
zeta_8ms = jnp.zeros(1)

xi_mon   = jnp.linspace(0.0, x_nd_max, 20)
eta_mon  = jnp.linspace(0.0, y_nd_max, 14)
zeta_mon = jnp.linspace(0.0, z_nd_max, 8)


# ══════════════════════════════════════════════════════════════════════════════
#  12.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
LOG_EVERY   = 500
best_loss   = np.inf
best_params = params
history     = {"total": [], "pde": [], "bc": [], "T_laser": [], "T_max": []}

print("-- Phase 1: Adam (warm-up + cosine decay) --")
t0 = time.time()

RESAMPLE_EVERY = 1

key, k1, k2 = jax.random.split(key, 3)
tau_c, xi_c, eta_c, zeta_c = sample_col(k1)
tau_b, xi_b, eta_b, zeta_b = sample_bc(k2)

for epoch in range(1, N_EPOCHS + 1):

    if epoch == N_EPOCHS_1 + 1:
        state = train_state.TrainState.create(
            apply_fn=apply_fn, params=state.params,
            tx=make_opt(LR_MIN))
        print(f"\n  ── Phase 2: fine-tune @ LR={LR_MIN:.1e} ──\n")

    if epoch % RESAMPLE_EVERY == 0:
        key, k1, k2 = jax.random.split(key, 3)
        tau_c, xi_c, eta_c, zeta_c = sample_col(k1)
        tau_b, xi_b, eta_b, zeta_b = sample_bc(k2)

    state, loss, (Lp, Lb) = train_step(
        state,
        tau_c, xi_c, eta_c, zeta_c,
        tau_b, xi_b, eta_b, zeta_b,
        w_pde, w_bc)

    lv, pv, bv = float(loss), float(Lp), float(Lb)

    history["total"].append(lv)
    history["pde"].append(pv)
    history["bc"].append(bv)

    if lv < best_loss:
        best_loss   = lv
        best_params = state.params

    if epoch % LOG_EVERY == 0 or epoch == 1:
        T_laser = float(T_phys(state.params,
                                tau_8ms, xi_8ms, eta_8ms, zeta_8ms)[0, 0, 0, 0])
        T_grid = np.array(T_phys(state.params, tau_8ms, xi_mon, eta_mon, zeta_mon))
        T_max  = float(T_grid.max())
        history["T_laser"].append(T_laser)
        history["T_max"].append(T_max)

        lr_now = float(lr_sched(epoch)) if epoch <= N_EPOCHS_1 else LR_MIN
        print(f"  Ep {epoch:6d}/{N_EPOCHS} | "
              f"L={lv:.3e}  PDE={pv:.3e}  BC={bv:.3e} | "
              f"LR={lr_now:.2e} | "
              f"T_laser={T_laser:.0f}K  T_max={T_max:.0f}K | "
              f"t={time.time()-t0:.0f}s")

print(f"\n  Done in {time.time()-t0:.1f}s  |  best loss = {best_loss:.4e}")
total_time = time.time() - t0
# ══════════════════════════════════════════════════════════════════════════════
#  13.  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def eval_xy(params, t_val, res=200):
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
    tau  = jnp.array([t_val / t_ref])
    xi   = jnp.linspace(0.0, x_nd_max, res)
    eta  = jnp.array([y_track / r_ref])
    zeta = jnp.linspace(0.0, z_nd_max, res)
    T    = np.array(T_phys(params, tau, xi, eta, zeta))[0, :, 0, :]
    X    = np.linspace(0, Lx * 1e3, res)
    Z    = np.linspace(0, Lz * 1e3, res)
    Xg, Zg = np.meshgrid(X, Z, indexing='ij')
    return Xg.T, Zg.T, T.T

def melt_pool(params, t_val, res=300):
    _, _, T_xy = eval_xy(params, t_val, res)
    _, _, T_xz = eval_xz(params, t_val, res)

    if T_xy.max() < T_melt:
        print("  ⚠ No melt pool detected (T_max < T_melt).")
        return 0.0, 0.0, 0.0

    X_mm = np.linspace(0, Lx * 1e3, res)
    Y_mm = np.linspace(0, Ly * 1e3, res)
    Z_mm = np.linspace(0, Lz * 1e3, res)

    dx_mm = X_mm[1] - X_mm[0]
    dy_mm = Y_mm[1] - Y_mm[0]
    dz_mm = Z_mm[1] - Z_mm[0]

    # Length
    length = 0.0
    for iy in range(res):
        low_end = 0.0
        high_end = 0.0
        for ix in range(res - 1):
            T_here = T_xy[iy, ix]
            T_next = T_xy[iy, ix + 1]
            if T_here < T_melt and T_next >= T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                low_end = X_mm[ix] + frac * dx_mm
            if T_here >= T_melt and T_next < T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                high_end = X_mm[ix] + frac * dx_mm
            if (high_end - low_end) > length:
                length = high_end - low_end

    # Width
    width = 0.0
    for ix in range(res):
        low_end = 0.0
        high_end = 0.0
        for iy in range(res - 1):
            T_here = T_xy[iy, ix]
            T_next = T_xy[iy + 1, ix]
            if T_here < T_melt and T_next >= T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                low_end = Y_mm[iy] + frac * dy_mm
            if T_here >= T_melt and T_next < T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                high_end = Y_mm[iy] + frac * dy_mm
            if (high_end - low_end) > width:
                width = high_end - low_end

    # Depth
    depth = 0.0
    for ix in range(res):
        for iz in range(res - 1):
            T_here = T_xz[iz, ix]
            T_next = T_xz[iz + 1, ix]
            if T_here >= T_melt and T_next < T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                d_local = Z_mm[iz] + frac * dz_mm
                if d_local > depth:
                    depth = d_local
            elif T_here < T_melt and T_next >= T_melt:
                frac = (T_melt - T_here) / (T_next - T_here)
                d_local = Z_mm[iz] + frac * dz_mm
                if d_local > depth:
                    depth = d_local

    print(f"  Melt pool: L={length:.3f} mm  W={width:.3f} mm  D={depth:.3f} mm")
    return length, width, depth

# ══════════════════════════════════════════════════════════════════════════════
#  14.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
CMAP     = "RdBu_r"
VMIN     = 400.0
VMAX     = 1600.0
INSET    = 100.0
N_LEVELS = 40
N_ISO    = 16
ISO_LW   = 0.5

Lx_mm = Lx * 1e3
Ly_mm = Ly * 1e3
Lz_mm = Lz * 1e3

def plot_snapshot(params, t_val, fname, res=200):
    """Two-panel figure: xy top surface + xz cross-section."""
    X_xy, Y_xy, T_xy = eval_xy(params, t_val, res)
    X_xz, Z_xz, T_xz = eval_xz(params, t_val, res)

    norm_cb  = Normalize(vmin=VMIN - INSET, vmax=VMAX + INSET)
    levels   = np.linspace(VMIN, VMAX, N_LEVELS + 1)
    iso_levs = np.linspace(VMIN, VMAX, N_ISO + 1)

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
        # ax.contour(X, Y, T, levels=iso_levs,
        #            colors="white", linewidths=ISO_LW, alpha=0.75)
        # if T.max() >= T_melt:
        #     ax.contour(X, Y, T, levels=[T_melt],
        #                colors="#FFD700", linewidths=2.0)
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

    n_T = len(history["T_laser"])
    if n_T > 0:
        ep_T = np.concatenate([[1], np.arange(LOG_EVERY, N_EPOCHS + 1, LOG_EVERY)])
        ep_T = ep_T[:n_T]

        axes[1].plot(ep_T, history["T_laser"], "#c0392b", lw=1.5,
                     label="T at laser centre")
        axes[1].plot(ep_T, history["T_max"],   "#e67e22", lw=1.5, ls="--",
                     label="T_max (coarse grid)")
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

spinn_save_path = os.path.join(RESULTS_DIR, "spinn_best.pkl")
with open(spinn_save_path, "wb") as f:
    pickle.dump(best_params, f)
print(f"  Saved best_params → {spinn_save_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  15.  GENERATE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
plot_losses(history, os.path.join(RESULTS_DIR, "spinn_v4_loss.pdf"))

print("-- Melt Pool Dimensions --")
for t_val, label in [(8e-3, "t08ms"), (t_end, "t10ms")]:
    print(f"\n  [{label}]  t = {t_val*1e3:.1f} ms")
    melt_pool(best_params, t_val)
    plot_snapshot(best_params, t_val,
                  os.path.join(RESULTS_DIR, f"spinn_v4_{label}.pdf"))

print(f"\n  Total training time : {total_time:.1f} s ({total_time/60:.2f} min)")
print("All done.")