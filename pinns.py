"""
Single-Track PINN for LPBF — Safari & Wessels (2025), Track 1
=================================================================
Parameters from Table 1:
  rho     = 8351.91 kg/m³
  eta*P   = 150.0 W,  r_beam = 450 µm,  v_scan = 0.1 m/s
  T0      = 300 K,  T_inf = 300 K,  h_conv = 10 W/(m²K)

Graph style: Safari & Wessels (2025) Fig. 7
  - RdBu_r colormap, white background
  - VERTICAL layout: xy panel on top, xz panel below
  - Axis labels and ticks shown (x/y/z in mm)
  - Aspect ratio proportional to physical domain dimensions
  - Fixed colour scale 400-1600 K, values outside are clipped
    to end colours (extend="neither"), NOT cut off
  - Single horizontal colorbar at bottom
  - np.ptp replaced with .max()-.min()
"""

import os
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Geometry & Process Parameters
# ─────────────────────────────────────────────────────────────────────────────
Lx = 2.0e-3          # [m]  domain length  (x, scan direction)
Ly = 1.5e-3          # [m]  domain width   (y)
Lz = 1.0e-3          # [m]  domain depth   (z)

y_track  = 0.50e-3   # [m]  Track 1 centreline
x0_laser = 0.50e-3   # [m]  laser start
x1_laser = 1.50e-3   # [m]  laser end

# ── Table 1 ──────────────────────────────────────────────────────────────────
rho     = 8351.91    # [kg/m³]
P_abs   = 150.0      # [W]       modified laser power (eta x P)
r_beam  = 450.0e-6   # [m]       beam radius
c_depth = 500.0e-6   # [m]       penetration depth
v_scan  = 0.10       # [m/s]     scan speed
T0      = 300.0      # [K]       initial temperature
T_inf   = 300.0      # [K]       ambient temperature
h_conv  = 10.0       # [W/(m²K)] convection on top surface (z = 0)
T_melt  = 1533.0     # [K]       Hastelloy X solidus
# ─────────────────────────────────────────────────────────────────────────────

t_end = (x1_laser - x0_laser) / v_scan   # 10 ms
print(f"Track 1 scan duration : {t_end*1e3:.1f} ms")
print(f"Laser x: {x0_laser*1e3:.1f} mm -> {x1_laser*1e3:.1f} mm")
print(f"Track 1 y-position    : {y_track*1e3:.2f} mm")
print(f"r_beam = {r_beam*1e6:.0f} um  |  P_abs = {P_abs:.1f} W")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Material Properties — Hastelloy X  (Safari Eq. 21)
# ─────────────────────────────────────────────────────────────────────────────
def kappa(T):
    """Thermal conductivity [W/(m·K)], T in Kelvin (torch tensor)."""
    return (229.87 + 0.0184 * T
            + 225.10 * torch.tanh(0.018 * (T - 1816.8)))

def cp_fn(T):
    """Apparent specific heat [J/(kg·K)] (torch tensor)."""
    return (407.62 + 0.142 * T
            - 61.43  * torch.exp(-3.1e-4 * (T - 798.0)**2)
            + 1054.96 * torch.exp(-6.2e-5 * (T - 1816.8)**2))

# Reference scales evaluated at T0
kap0 = float(kappa(torch.tensor(T0)).item())
cp0  = float(cp_fn(torch.tensor(T0)).item())
alp0 = kap0 / (rho * cp0)
print(f"\nAt T0={T0:.0f} K:  k0={kap0:.2f} W/(m·K)  |  "
      f"cp0={cp0:.1f} J/(kg·K)  |  a0={alp0:.3e} m²/s")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Heat Source  (Goldak hemispherical Gaussian)
# ─────────────────────────────────────────────────────────────────────────────
qv_peak = (6.0 * np.sqrt(3.0) * P_abs) / (
           np.pi * np.sqrt(np.pi) * r_beam**2 * c_depth)
print(f"qv_peak               : {qv_peak:.4e} W/m³\n")

def heat_source(x_p, y_p, z_p, t_p):
    """Volumetric heat source [W/m³]. All inputs are physical torch tensors."""
    x_laser = torch.clamp(x0_laser + v_scan * t_p, max=float(x1_laser))
    return qv_peak * (
        torch.exp(-3.0 * ((x_p - x_laser)**2 + (y_p - y_track)**2) / r_beam**2)
        * torch.exp(-3.0 * (z_p**2 / c_depth**2))
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Non-dimensionalisation
# ─────────────────────────────────────────────────────────────────────────────
r_ref  = r_beam
t_ref  = r_beam**2 / alp0
DT_ref = qv_peak * r_beam**2 / kap0

print(f"Non-dim scales:")
print(f"  r_ref  = {r_ref*1e6:.0f} um")
print(f"  t_ref  = {t_ref*1e3:.4f} ms")
print(f"  DT_ref = {DT_ref:.1f} K  ->  T_peak_est ~ {T0+DT_ref:.0f} K")
print(f"  t_end_nd = {t_end/t_ref:.4f}\n")

x_nd_min = 0.0;  x_nd_max = Lx / r_ref
y_nd_min = 0.0;  y_nd_max = Ly / r_ref
z_nd_min = 0.0;  z_nd_max = Lz / c_depth
t_nd_min = 0.0;  t_nd_max = t_end / t_ref

x0_nd = x0_laser / r_ref
x1_nd = x1_laser / r_ref
yt_nd = y_track  / r_ref

def nd(x_p, y_p, z_p, t_p):
    """Physical [m,m,m,s] -> non-dimensional."""
    return x_p/r_ref, y_p/r_ref, z_p/c_depth, t_p/t_ref

def to_net(x_nd, y_nd, z_nd, t_nd):
    """Non-dim -> [-1, 1] for network input."""
    def sc(v, lo, hi): return 2.0*(v - lo)/(hi - lo) - 1.0
    return (sc(x_nd, x_nd_min, x_nd_max),
            sc(y_nd, y_nd_min, y_nd_max),
            sc(z_nd, z_nd_min, z_nd_max),
            sc(t_nd, t_nd_min, t_nd_max))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Network
# ─────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def T_hat(self, x_nd, y_nd, z_nd, t_nd):
        """Dimensionless temperature rise T_hat >= 0 (softplus)."""
        xn, yn, zn, tn = to_net(x_nd, y_nd, z_nd, t_nd)
        inp = torch.cat([xn, yn, zn, tn], dim=1)
        return torch.nn.functional.softplus(self.net(inp))

    def T_phys(self, x_nd, y_nd, z_nd, t_nd):
        """Physical temperature [K]."""
        return T0 + self.T_hat(x_nd, y_nd, z_nd, t_nd) * DT_ref

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Tensor helpers
# ─────────────────────────────────────────────────────────────────────────────
def ten(a, grad=False):
    return (torch.tensor(np.asarray(a, dtype=np.float32), device=device)
            .reshape(-1, 1).requires_grad_(grad))

def grad1(out, inp):
    return torch.autograd.grad(
        out, inp,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True)[0]

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PDE Loss
# ─────────────────────────────────────────────────────────────────────────────
def pde_loss_fn(model, x_nd, y_nd, z_nd, t_nd):
    Th  = model.T_hat(x_nd, y_nd, z_nd, t_nd)
    T_K = T0 + Th * DT_ref

    dTh_dt   = grad1(Th, t_nd)
    dTh_dx   = grad1(Th, x_nd)
    dTh_dy   = grad1(Th, y_nd)
    dTh_dz   = grad1(Th, z_nd)
    d2Th_dx2 = grad1(dTh_dx, x_nd)
    d2Th_dy2 = grad1(dTh_dy, y_nd)
    d2Th_dz2 = grad1(dTh_dz, z_nd)

    k     = kappa(T_K)
    c     = cp_fn(T_K)
    dkdTh = grad1(k, Th)
    dkdT  = dkdTh / DT_ref

    dT_dt   = (DT_ref / t_ref)      * dTh_dt
    d2T_dx2 = (DT_ref / r_ref**2)   * d2Th_dx2
    d2T_dy2 = (DT_ref / r_ref**2)   * d2Th_dy2
    d2T_dz2 = (DT_ref / c_depth**2) * d2Th_dz2
    dT_dx   = (DT_ref / r_ref)      * dTh_dx
    dT_dy   = (DT_ref / r_ref)      * dTh_dy
    dT_dz   = (DT_ref / c_depth)    * dTh_dz

    div_kgradT = (k * (d2T_dx2 + d2T_dy2 + d2T_dz2)
                  + dkdT * (dT_dx**2 + dT_dy**2 + dT_dz**2))

    x_p = x_nd * r_ref;   y_p = y_nd * r_ref
    z_p = z_nd * c_depth;  t_p = t_nd * t_ref
    qv  = heat_source(x_p, y_p, z_p, t_p)

    R = (rho * c * dT_dt - div_kgradT - qv) / qv_peak
    return torch.mean(R**2)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  IC Loss
# ─────────────────────────────────────────────────────────────────────────────
def ic_loss_fn(model, n=2048):
    x = np.random.uniform(x_nd_min, x_nd_max, n)
    y = np.random.uniform(y_nd_min, y_nd_max, n)
    z = np.random.uniform(z_nd_min, z_nd_max, n)
    t = np.zeros(n)
    Th = model.T_hat(ten(x), ten(y), ten(z), ten(t))
    return torch.mean(Th**2)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  BC Loss
# ─────────────────────────────────────────────────────────────────────────────
def bc_loss_fn(model, n=512):
    t_rnd = np.random.uniform(t_nd_min, t_nd_max, n)
    loss  = torch.tensor(0.0, device=device)

    # Top surface z=0: convection BC
    x_top = np.random.uniform(x_nd_min, x_nd_max, n)
    y_top = np.random.uniform(y_nd_min, y_nd_max, n)
    z_top = ten(np.zeros(n), grad=True)

    Th_top     = model.T_hat(ten(x_top), ten(y_top), z_top, ten(t_rnd))
    T_K_top    = T0 + Th_top * DT_ref
    dTh_dz_top = grad1(Th_top, z_top)
    k_top      = kappa(T_K_top)

    res_conv = (k_top / c_depth * dTh_dz_top * DT_ref
                - h_conv * (T_K_top - T_inf))
    loss = loss + torch.mean(res_conv**2) / (h_conv * DT_ref)**2

    # Five insulated faces
    insulated_faces = [
        ("x", x_nd_min, (y_nd_min, y_nd_max), (z_nd_min, z_nd_max)),
        ("x", x_nd_max, (y_nd_min, y_nd_max), (z_nd_min, z_nd_max)),
        ("y", y_nd_min, (x_nd_min, x_nd_max), (z_nd_min, z_nd_max)),
        ("y", y_nd_max, (x_nd_min, x_nd_max), (z_nd_min, z_nd_max)),
        ("z", z_nd_max, (x_nd_min, x_nd_max), (y_nd_min, y_nd_max)),
    ]

    for fixed, val, r1, r2 in insulated_faces:
        f1 = np.random.uniform(r1[0], r1[1], n)
        f2 = np.random.uniform(r2[0], r2[1], n)

        if fixed == "x":
            xb = ten(np.full(n, val), grad=True)
            yb = ten(f1); zb = ten(f2); wrt = xb
        elif fixed == "y":
            yb = ten(np.full(n, val), grad=True)
            xb = ten(f1); zb = ten(f2); wrt = yb
        else:
            zb = ten(np.full(n, val), grad=True)
            xb = ten(f1); yb = ten(f2); wrt = zb

        Th   = model.T_hat(xb, yb, zb, ten(t_rnd))
        dTdn = grad1(Th, wrt)
        loss = loss + torch.mean(dTdn**2)

    return loss / (len(insulated_faces) + 1)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Collocation Sampling
# ─────────────────────────────────────────────────────────────────────────────
def sample_collocation(n_sobol=4096, n_laser=4096):
    n_sobol = 1 << (n_sobol - 1).bit_length()
    sampler = qmc.Sobol(d=4, scramble=True)
    raw     = sampler.random(n_sobol)
    xs = raw[:,0]*(x_nd_max-x_nd_min) + x_nd_min
    ys = raw[:,1]*(y_nd_max-y_nd_min) + y_nd_min
    zs = raw[:,2]*(z_nd_max-z_nd_min) + z_nd_min
    ts = raw[:,3]*(t_nd_max-t_nd_min) + t_nd_min

    t_l = np.random.uniform(t_nd_min, t_nd_max, n_laser)
    x_laser_nd = np.clip(x0_nd + (v_scan * t_l * t_ref) / r_ref,
                         x0_nd, x1_nd)
    xl = np.clip(x_laser_nd + np.random.uniform(-2, 2, n_laser),
                 x_nd_min, x_nd_max)
    yl = np.clip(yt_nd + np.random.uniform(-2, 2, n_laser),
                 y_nd_min, y_nd_max)
    zl = np.random.uniform(0, 2.0, n_laser)
    tl = t_l

    return (np.concatenate([xs, xl]),
            np.concatenate([ys, yl]),
            np.concatenate([zs, zl]),
            np.concatenate([ts, tl]))

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Training
# ─────────────────────────────────────────────────────────────────────────────
def train(adam_steps=20000, lbfgs_steps=500,
          w_pde=1.0, w_ic=1.0, w_bc=0.1,
          n_sobol=4096, n_laser=4096):

    model = PINN(width=128, depth=6).to(device)

    xi, yi, zi, ti = sample_collocation(n_sobol, n_laser)
    x_col = ten(xi, grad=True)
    y_col = ten(yi, grad=True)
    z_col = ten(zi, grad=True)
    t_col = ten(ti, grad=True)

    hist = {"total": [], "pde": [], "ic": [], "bc": []}

    def losses():
        Lp = pde_loss_fn(model, x_col, y_col, z_col, t_col)
        Li = ic_loss_fn(model)
        Lb = bc_loss_fn(model)
        L  = w_pde*Lp + w_ic*Li + w_bc*Lb
        return L, Lp, Li, Lb

    # ── Adam ──────────────────────────────────────────────────────────────
    print("-- Adam warm-up --")
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=2000, T_mult=2, eta_min=1e-5)

    best_loss  = float("inf")
    best_state = None

    for step in range(1, adam_steps + 1):
        opt.zero_grad()
        L, Lp, Li, Lb = losses()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        lv = L.item()
        hist["total"].append(lv)
        hist["pde"].append(Lp.item())
        hist["ic"].append(Li.item())
        hist["bc"].append(Lb.item())

        if lv < best_loss:
            best_loss  = lv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if step % 500 == 0 or step == 1:
            print(f"  Step {step:6d} | lr={opt.param_groups[0]['lr']:.2e} | "
                  f"Total:{lv:.3e} | PDE:{Lp.item():.3e} | "
                  f"IC:{Li.item():.3e} | BC:{Lb.item():.3e}")

    model.load_state_dict(best_state)
    print(f"\n  Best Adam loss: {best_loss:.3e}")

    # ── L-BFGS ────────────────────────────────────────────────────────────
    print("\n-- L-BFGS fine-tuning --")
    opt2 = torch.optim.LBFGS(
        model.parameters(), max_iter=lbfgs_steps,
        history_size=100, tolerance_grad=1e-9,
        tolerance_change=1e-11, line_search_fn="strong_wolfe")

    it = [0]
    def closure():
        opt2.zero_grad()
        L, Lp, Li, Lb = losses()
        L.backward()
        hist["total"].append(L.item())
        hist["pde"].append(Lp.item())
        hist["ic"].append(Li.item())
        hist["bc"].append(Lb.item())
        if it[0] % 50 == 0:
            print(f"  LBFGS {it[0]:4d} | Total:{L.item():.3e} | "
                  f"PDE:{Lp.item():.3e} | IC:{Li.item():.3e} | "
                  f"BC:{Lb.item():.3e}")
        it[0] += 1
        return L

    opt2.step(closure)
    print(f"\n  Final loss: {hist['total'][-1]:.3e}")
    return model, hist

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_surface(model, t_phys, res=150):
    """Top surface (z=0) temperature map. Returns X[mm], Y[mm], T[K]."""
    model.eval()
    xs_p = np.linspace(0, Lx, res)
    ys_p = np.linspace(0, Ly, res)
    X_p, Y_p = np.meshgrid(xs_p, ys_p)
    xn, yn, zn, tn = nd(X_p.ravel(), Y_p.ravel(),
                         np.zeros(res*res), np.full(res*res, t_phys))
    T = model.T_phys(ten(xn), ten(yn), ten(zn), ten(tn))
    return X_p*1e3, Y_p*1e3, T.cpu().numpy().reshape(res, res)

@torch.no_grad()
def eval_xz_plane(model, t_phys, res=150):
    """xz cross-section at y=y_track. Returns X[mm], Z[mm], T[K]."""
    model.eval()
    xs_p = np.linspace(0, Lx, res)
    zs_p = np.linspace(0, Lz, res)
    X_p, Z_p = np.meshgrid(xs_p, zs_p)
    xn, yn, zn, tn = nd(X_p.ravel(), np.full(res*res, y_track),
                         Z_p.ravel(), np.full(res*res, t_phys))
    T = model.T_phys(ten(xn), ten(yn), ten(zn), ten(tn))
    return X_p*1e3, Z_p*1e3, T.cpu().numpy().reshape(res, res)

def get_melt_pool(model, t_phys, res=60):
    """Melt pool L, W, D [mm]. Uses .max()-.min() — no deprecation."""
    model.eval()
    xs_p = np.linspace(0, Lx, res)
    ys_p = np.linspace(0, Ly, res)
    zs_p = np.linspace(0, Lz, res)
    T_vol = np.zeros((res, res, res))

    with torch.no_grad():
        for i, xi in enumerate(xs_p):
            yg, zg = np.meshgrid(ys_p, zs_p, indexing="ij")
            n_pts  = res * res
            xn, yn, zn, tn = nd(
                np.full(n_pts, xi), yg.ravel(),
                zg.ravel(), np.full(n_pts, t_phys))
            T = model.T_phys(ten(xn), ten(yn), ten(zn), ten(tn))
            T_vol[i] = T.cpu().numpy().reshape(res, res)

    print(f"  T range: [{T_vol.min():.1f}, {T_vol.max():.1f}] K")
    melt = T_vol > T_melt

    if not melt.any():
        print("  No melt pool detected (T_max < T_melt).")
        return 0.0, 0.0, 0.0

    x_melt = xs_p[melt.any(axis=(1, 2))]
    y_melt = ys_p[melt.any(axis=(0, 2))]
    z_melt = zs_p[melt.any(axis=(0, 1))]

    L = float(x_melt.max() - x_melt.min()) * 1e3
    W = float(y_melt.max() - y_melt.min()) * 1e3
    D = float(z_melt.max() - z_melt.min()) * 1e3

    print(f"  Melt Pool -> L={L:.3f} mm | W={W:.3f} mm | D={D:.3f} mm")
    return L, W, D

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Plot Style Constants  (Safari & Wessels 2025, Fig. 7)
# ─────────────────────────────────────────────────────────────────────────────
CMAP      = "RdBu_r"   # blue=cold -> white=mid -> red=hot
VMIN      = 400.0      # [K] fixed colour scale lower bound
VMAX      = 1600.0     # [K] fixed colour scale upper bound
N_LEVELS  = 40         # filled contour bands
N_ISO     = 16         # white iso-contour lines
ISO_LW    = 0.5
MELT_LW   = 1.4

# Physical domain sizes in mm (used for aspect ratio)
Lx_mm = Lx * 1e3      # 2.0 mm
Ly_mm = Ly * 1e3      # 1.5 mm
Lz_mm = Lz * 1e3      # 1.0 mm

def _style_ax(ax, xlabel, ylabel, xlim, ylim):
    """
    Apply clean paper-style formatting WITH axis labels and ticks.
    """
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=9, direction="in",
                   top=True, right=True, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
    ax.set_facecolor("white")

def _draw_contours(ax, X, Y, T):
    """
    Filled contours (clipped to VMIN-VMAX, not cut off) +
    white iso-lines + melt pool boundary.
    Key: extend="neither" clips values outside range to end colours.
    """
    levels = np.linspace(VMIN, VMAX, N_LEVELS + 1)
    iso_lv = np.linspace(VMIN, VMAX, N_ISO + 1)

    # Clip data to [VMIN, VMAX] so areas outside still show end colours
    T_clipped = np.clip(T, VMIN, VMAX)

    ax.contourf(X, Y, T_clipped, levels=levels,
                vmin=VMIN, vmax=VMAX,
                cmap=CMAP, extend="neither")

    # White iso-contour lines on ORIGINAL (unclipped) data
    ax.contour(X, Y, T, levels=iso_lv,
               colors="white", linewidths=ISO_LW, alpha=0.75)

    # Melt pool boundary
    if T.max() > T_melt:
        ax.contour(X, Y, T, levels=[T_melt],
                   colors="white", linewidths=MELT_LW)

def _add_colorbar(fig, ax_list):
    """
    Single horizontal colorbar below all panels.
    Ticks at 400, 800, 1200, 1600 K — exactly like the paper.
    """
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    sm   = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax_list,
        orientation="horizontal",
        fraction=0.04,
        pad=0.12,
        aspect=35,
        shrink=0.55)
    cbar.set_label("Temperature (K)", fontsize=11)
    cbar.set_ticks([400, 800, 1200, 1600])
    cbar.ax.tick_params(labelsize=10)
    return cbar

# ─────────────────────────────────────────────────────────────────────────────
# 14.  Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_snapshot(model, t_phys, fname="snapshot.png", res=150):
    """
    VERTICAL two-panel figure matching Safari & Wessels (2025) Fig. 7:
      TOP    : xy top surface (z = 0)
      BOTTOM : xz cross-section at y = y_track

    - No suptitle
    - Increased gap between panels
    - Colorbar norm wider than [400,1600] so bar fills fully to edges
    - Axis coordinates shown, no laser line, no melt pool contour
    """
    X_xy, Y_xy, T_xy = eval_surface(model, t_phys, res)
    X_xz, Z_xz, T_xz = eval_xz_plane(model, t_phys, res)

    T_max_all = max(float(T_xy.max()), float(T_xz.max()))
    T_min_all = min(float(T_xy.min()), float(T_xz.min()))

    print(f"  T range : {T_min_all:.0f} K -> {T_max_all:.0f} K")

    # ── Layout ────────────────────────────────────────────────────────────
    panel_w_in = 7.0
    panel_ar   = 2.5
    panel_h_in = panel_w_in / panel_ar      # ~2.8 in

    gap_in     = 1.00    # ← increased from 0.55 for more breathing room
    top_m_in   = 0.25    # ← reduced: no suptitle so less top margin needed
    bot_m_in   = 1.80
    left_m_in  = 0.70
    right_m_in = 0.25

    fig_w = left_m_in + panel_w_in + right_m_in
    fig_h = top_m_in + panel_h_in + gap_in + panel_h_in + bot_m_in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def to_frac(left_in, bottom_in, w_in, h_in):
        return [left_in/fig_w, bottom_in/fig_h, w_in/fig_w, h_in/fig_h]

    xz_bot_in = bot_m_in
    xy_bot_in = bot_m_in + panel_h_in + gap_in

    ax_xy = fig.add_axes(to_frac(left_m_in, xy_bot_in, panel_w_in, panel_h_in))
    ax_xz = fig.add_axes(to_frac(left_m_in, xz_bot_in, panel_w_in, panel_h_in))

    # ── Panel drawing helper ──────────────────────────────────────────────
    def _draw_panel(ax, X, Y, T, title, xlabel, ylabel, ylim, invert_y=False):

        levels = np.linspace(VMIN, VMAX, N_LEVELS + 1)
        iso_lv = np.linspace(VMIN, VMAX, N_ISO + 1)

        T_cl = np.clip(T, VMIN, VMAX)

        ax.contourf(X, Y, T_cl,
                    levels=levels, vmin=VMIN, vmax=VMAX,
                    cmap=CMAP, extend="neither")

        # White iso-lines on original unclipped data
        ax.contour(X, Y, T,
                   levels=iso_lv,
                   colors="white", linewidths=ISO_LW, alpha=0.75)

        # NO laser line. NO melt pool contour.

        if invert_y:
            ax.invert_yaxis()

        ax.set_xlim(0, Lx_mm)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9, direction="in",
                       top=True, right=True, length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=10, pad=5, loc="center")

    # ── Draw panels ───────────────────────────────────────────────────────
    _draw_panel(
        ax_xy, X_xy, Y_xy, T_xy,
        title    = f"xy-plane  (z = 0, top surface)   t = {t_phys*1e3:.1f} ms",
        xlabel   = "x [mm]",
        ylabel   = "y [mm]",
        ylim     = (0, Ly_mm),
        invert_y = False
    )
    _draw_panel(
        ax_xz, X_xz, Z_xz, T_xz,
        title    = f"xz-plane  (y = {y_track*1e3:.2f} mm, Track 1)   t = {t_phys*1e3:.1f} ms",
        xlabel   = "x [mm]",
        ylabel   = "z [mm]",
        ylim     = (Lz_mm, 0),
        invert_y = True
    )

    # ── Colorbar ──────────────────────────────────────────────────────────
    cbar_left_in = left_m_in + panel_w_in * 0.08
    cbar_w_in    = panel_w_in * 0.84
    cbar_h_in    = 0.30
    cbar_bot_in  = bot_m_in * 0.30

    cax = fig.add_axes(to_frac(cbar_left_in, cbar_bot_in,
                                cbar_w_in, cbar_h_in))

    # Norm wider than [400,1600] → colourmap fills fully to bar edges,
    # ticks at 400/800/1200/1600 sit naturally inset — matches reference.
    inset   = 100.0
    norm_cb = Normalize(vmin=VMIN - inset, vmax=VMAX + inset)
    sm      = cm.ScalarMappable(cmap=CMAP, norm=norm_cb)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal",
                        extend="neither")

    cbar.set_ticks([400, 800, 1200, 1600])
    cbar.ax.tick_params(
        labelsize   = 14,
        length      = 7,
        width       = 1.2,
        direction   = "out",
        bottom      = True,
        top         = False,
        labelbottom = True,
        labeltop    = False
    )
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor("black")
    cbar.set_label("Temperature (K)", fontsize=14, labelpad=8)

    # ── NO suptitle ───────────────────────────────────────────────────────

    plt.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

def plot_centreline(model, t_phys, fname="centreline.png"):
    """T vs x along laser centreline (y=y_track, z=0)."""
    model.eval()
    xs_p = np.linspace(0, Lx, 400)
    xn, yn, zn, tn = nd(xs_p, np.full(400, y_track),
                         np.zeros(400), np.full(400, t_phys))
    with torch.no_grad():
        T = model.T_phys(ten(xn), ten(yn), ten(zn), ten(tn))
    T = T.cpu().numpy().ravel()
    x_laser_mm = min(x0_laser + v_scan * t_phys, x1_laser) * 1e3

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(xs_p * 1e3, T, color="#c0392b", lw=2.0, label="PINN")
    ax.axhline(T_melt, color="#2980b9", ls="--", lw=1.5,
               label=f"$T_{{solidus}}$ = {T_melt:.0f} K")
    ax.axhline(T0, color="gray", ls=":", lw=1.0,
               label=f"$T_0$ = {T0:.0f} K")
    ax.axvline(x_laser_mm, color="black", ls="--", lw=1.2, alpha=0.6,
               label=f"Laser @ {x_laser_mm:.2f} mm")
    ax.axvspan(x0_laser * 1e3, x1_laser * 1e3,
               alpha=0.06, color="#e67e22", label="Scan region")
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel("T [K]", fontsize=12)
    ax.set_title(
        f"Centreline Temperature  (y = {y_track*1e3:.2f} mm, z = 0)  "
        f"t = {t_phys*1e3:.1f} ms", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, color="gray", linestyle="--")
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

def plot_losses(hist, fname="loss_history.png"):
    n_adam = 20000
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.semilogy(hist["total"], color="black",   lw=1.8, label="Total")
    ax.semilogy(hist["pde"],   color="#c0392b", lw=1.2, label="PDE")
    ax.semilogy(hist["ic"],    color="#2980b9", lw=1.2, ls="--", label="IC")
    ax.semilogy(hist["bc"],    color="#27ae60", lw=1.2, ls=":",  label="BC")
    ax.axvline(n_adam, color="orange", ls=":", lw=1.5,
               label="Adam -> L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss — Track 1 PINN", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25, color="gray", linestyle="--")
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 15.  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("  PINN — Track 1  (Safari & Wessels 2025, Table 1)")
    print(f"  Domain  : {Lx*1e3:.1f} x {Ly*1e3:.1f} x {Lz*1e3:.1f} mm")
    print(f"  Track 1 : y = {y_track*1e3:.2f} mm")
    print(f"  Laser   : x = {x0_laser*1e3:.1f} -> {x1_laser*1e3:.1f} mm")
    print(f"  r_beam  : {r_beam*1e6:.0f} um  |  c_depth: {c_depth*1e6:.0f} um")
    print(f"  P_abs   : {P_abs:.1f} W  |  v_scan: {v_scan*1e3:.0f} mm/s")
    print(f"  rho     : {rho:.2f} kg/m3")
    print(f"  T0={T0:.0f} K  |  T_inf={T_inf:.0f} K  |  h_conv={h_conv:.0f} W/(m2K)")
    print(f"  T_melt  : {T_melt:.0f} K  (Hastelloy X solidus)")
    print("=" * 60)

    model, hist = train(
        adam_steps  = 20000,
        lbfgs_steps = 500,
        w_pde = 1.0,
        w_ic  = 1.0,
        w_bc  = 0.1,
        n_sobol = 524288,
        n_laser = 32768,
    )

    torch.save(model.state_dict(), "results/pinn_track1.pt")
    print("  Saved: results/pinn_track1.pt")
    plot_losses(hist, "results/loss_history.png")

    snapshots = {"t08ms": 8e-3, "t10ms": t_end}

    print("\n-- Melt Pool Dimensions --")
    for label, t_q in snapshots.items():
        x_las = min(x0_laser + v_scan * t_q, x1_laser) * 1e3
        print(f"\n{label}  (laser @ x={x_las:.2f} mm):")
        get_melt_pool(model, t_q, res=60)
        plot_snapshot(model, t_q,
                      fname=f"results/snapshot_{label}.png", res=150)
        plot_centreline(model, t_q,
                        fname=f"results/centreline_{label}.png")

    print("\nAll done.")