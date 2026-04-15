import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Output path
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

#  True parameters θ*
KAPPA   = 2.0
THETA   = 0.04      # long-run variance  (= 4%)
SIGMA_V = 0.30
V0      = 0.04      # initial variance   (= 4%)
T_END   = 1.0       # simulate over one year
N_STEPS = 500       # time steps
N_PATHS = 5

#  Plot style
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Five distinct, clearly separable path colours
PATH_COLORS = ["#2166AC",   # blue
               "#1B7837",   # green
               "#D6604D",   # red
               "#762A83",   # purple
               "#F46D43"]   # orange

BAND_COLOR = "steelblue"

#  SIMULATE CIR VARIANCE PATHS
#  Euler full-truncation scheme  (Lord et al. 2010)
#
#  V_{t+Δt} = V_t + κ(θ − V_t^+)Δt + σ_v √(V_t^+ Δt) Z
#
#  V_t^+ = max(V_t, 0)  ← full truncation prevents negative variance
#  All values stored and plotted as variance (NOT volatility),
#  scaled to percentage points by multiplying by 100.


dt     = T_END / N_STEPS
t_grid = np.linspace(0.0, T_END, N_STEPS + 1)

np.random.seed(7)   # fixed seed for reproducibility
paths = np.zeros((N_PATHS, N_STEPS + 1))
paths[:, 0] = V0

for step in range(N_STEPS):
    Vp          = np.maximum(paths[:, step], 0.0)       # full truncation
    Z           = np.random.randn(N_PATHS)
    paths[:, step + 1] = (
        paths[:, step]
        + KAPPA * (THETA - Vp) * dt
        + SIGMA_V * np.sqrt(Vp * dt) * Z
    )

# Convert variance to percentage points (multiply by 100)
# e.g. V = 0.04  →  4%
paths_pct = paths * 100.0

# Analytical ±1 SD band for the CIR process
# E[V_t]   = θ + (V_0 − θ) e^{−κt}
# Var[V_t] = V_0 σ_v² e^{−κt}/κ · (1 − e^{−κt})
#           + θ σ_v²/(2κ) · (1 − e^{−κt})²
e_v   = THETA + (V0 - THETA) * np.exp(-KAPPA * t_grid)
var_v = (V0 * SIGMA_V ** 2 * np.exp(-KAPPA * t_grid) / KAPPA
         * (1.0 - np.exp(-KAPPA * t_grid))
         + THETA * SIGMA_V ** 2 / (2.0 * KAPPA)
         * (1.0 - np.exp(-KAPPA * t_grid)) ** 2)
std_v = np.sqrt(np.maximum(var_v, 0.0))

# Convert to percentage points
e_v_pct   = e_v   * 100.0
std_v_pct = std_v * 100.0
theta_pct = THETA * 100.0          # 4%


#  PLOT
fig, ax = plt.subplots(figsize=(9, 5))

# ±1 SD band (analytical)
ax.fill_between(
    t_grid,
    e_v_pct - std_v_pct,
    e_v_pct + std_v_pct,
    alpha=0.15, color=BAND_COLOR,
    label=r"Approx. $\pm 1$ SD band"
)

# Five individual paths
for k in range(N_PATHS):
    ax.plot(t_grid, paths_pct[k],
            color=PATH_COLORS[k], lw=1.2, alpha=0.90,
            label=f"Path {k + 1}")

# Long-run level dashed line
ax.axhline(
    theta_pct,
    color="black", lw=1.8, ls="--",
    label=r"Long-run level $\theta^* = 4\%$"
)

# Labels and formatting
ax.set_xlabel("Time (years)")
ax.set_ylabel(r"Variance $V_t$ (×100, i.e. percentage points)")
ax.set_title(
    "Sample Paths of the CIR Variance Process\n"
    r"($\kappa = 2.0,\ \theta = 0.04,\ \sigma_v = 0.30$,"
    r"  Feller condition satisfied: $2\kappa\theta = 0.16 > \sigma_v^2 = 0.09$)"
)

# Annotation box explaining the y-axis scale
ax.text(
    0.02, 0.97,
    r"Note: $V_t = 0.04$ plots as $4$ on this axis",
    transform=ax.transAxes,
    fontsize=8.5, va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
              edgecolor="grey", alpha=0.8)
)

ax.set_xlim(0.0, T_END)
ax.set_ylim(bottom=0.0)

# Legend: paths first, then dashed line, then band
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right",
          fontsize=8.5, framealpha=0.9,
          ncol=2)

plt.tight_layout()

#  SAVE
for ext in ("pdf", "png"):
    fpath = os.path.join(SAVE_DIR, f"fig67_var_paths.{ext}")
    plt.savefig(fpath, bbox_inches="tight", dpi=300)
    print(f"Saved: {fpath}")

plt.close()
print("Done.")
