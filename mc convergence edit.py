import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ── Output path ───────────────────────────────────────────────────────────────
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── True parameters θ* ────────────────────────────────────────────────────────
KAPPA   = 2.0
THETA   = 0.04
SIGMA_V = 0.30
RHO     = -0.70
V0      = 0.04
R       = 0.05
S0      = 100.0
K_ATM   = 100.0
TAU     = 1.0          # 1-year ATM call
N_STEPS = 200          # time steps per path

# ── Plot style ────────────────────────────────────────────────────────────────
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

BLUE = "#2166AC"
RED  = "#D6604D"

# ════════════════════════════════════════════════════════════════════════════
#  SEMI-ANALYTICAL HESTON PRICE  (Albrecher et al. branch-cut-safe form)
# ════════════════════════════════════════════════════════════════════════════

def heston_cf(u, S, v, tau, kappa, theta, sigma_v, rho, r):
    """
    Heston characteristic function of ln(S_T) under Q.
    Uses the Albrecher et al. (2007) branch-cut-safe reparameterisation.
    """
    i   = 1j
    d   = np.sqrt((kappa - i * rho * sigma_v * u) ** 2
                  + sigma_v ** 2 * (u ** 2 + i * u))
    b   = kappa - i * rho * sigma_v * u
    g   = (b - d) / (b + d)
    D   = ((b - d) / sigma_v ** 2
           * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau)))
    C   = (i * u * r * tau
           + kappa * theta / sigma_v ** 2
             * ((b - d) * tau
                - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))))
    return np.exp(C + D * v + i * u * np.log(S))


def heston_price_sa(S, v, K, tau, kappa, theta, sigma_v, rho, r):
    """
    Semi-analytical Heston European call price via Gil-Pelaez inversion.
    Integration limits [1e-8, 200], tolerance 1e-8.
    """
    i = 1j

    def integrand_P1(u):
        cf  = heston_cf(u - i, S, v, tau, kappa, theta, sigma_v, rho, r)
        cf0 = heston_cf(-i,    S, v, tau, kappa, theta, sigma_v, rho, r)
        phi1 = cf / cf0
        return np.real(np.exp(-i * u * np.log(K)) * phi1 / (i * u))

    def integrand_P2(u):
        phi2 = heston_cf(u, S, v, tau, kappa, theta, sigma_v, rho, r)
        return np.real(np.exp(-i * u * np.log(K)) * phi2 / (i * u))

    P1, _ = quad(integrand_P1, 1e-8, 200, limit=200, epsabs=1e-8)
    P2, _ = quad(integrand_P2, 1e-8, 200, limit=200, epsabs=1e-8)
    P1 = 0.5 + P1 / np.pi
    P2 = 0.5 + P2 / np.pi
    return S * P1 - K * np.exp(-r * tau) * P2


# ════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO WITH ANTITHETIC VARIATES
#  Euler full-truncation scheme  (Lord et al. 2010)
# ════════════════════════════════════════════════════════════════════════════

def mc_heston_antithetic(S0, V0, K, tau, kappa, theta, sigma_v, rho, r,
                          M, N, seed=42):
    """
    Price a European call using Heston Euler full-truncation MC
    with antithetic variates.

    Returns
    -------
    price : float   – antithetic MC price estimate
    se    : float   – standard error of the estimate
    """
    np.random.seed(seed)
    dt  = tau / N

    # Independent standard normals: shape (M, N)
    Z1 = np.random.randn(M, N)
    Z2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * np.random.randn(M, N)

    def simulate_paths(z1, z2):
        """Simulate M asset paths; return terminal stock prices."""
        V = np.full(M, V0, dtype=float)
        S = np.full(M, S0, dtype=float)
        for t in range(N):
            Vp  = np.maximum(V, 0.0)          # full truncation
            V   = (V
                   + kappa * (theta - Vp) * dt
                   + sigma_v * np.sqrt(Vp * dt) * z2[:, t])
            S   = S * np.exp(
                      (r - 0.5 * Vp) * dt
                      + np.sqrt(Vp * dt) * z1[:, t]
                  )
        return S

    # Original paths
    S_T   = simulate_paths(Z1,  Z2)
    # Antithetic paths  (negate both Brownians)
    S_T_a = simulate_paths(-Z1, -Z2)

    discount  = np.exp(-r * tau)
    payoff    = discount * (
        0.5 * np.maximum(S_T   - K, 0.0)
      + 0.5 * np.maximum(S_T_a - K, 0.0)
    )

    price = payoff.mean()
    se    = payoff.std(ddof=1) / np.sqrt(M)
    return price, se


# ════════════════════════════════════════════════════════════════════════════
#  COMPUTE RESULTS
# ════════════════════════════════════════════════════════════════════════════

print("Computing semi-analytical benchmark price...")
SA_PRICE = heston_price_sa(S0, V0, K_ATM, TAU,
                            KAPPA, THETA, SIGMA_V, RHO, R)
print(f"  Semi-analytical price = {SA_PRICE:.4f}")

PATH_COUNTS = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000])

print("Running Monte Carlo across path counts (antithetic variates)...")
mc_prices = []
mc_ses    = []
for M in PATH_COUNTS:
    price, se = mc_heston_antithetic(S0, V0, K_ATM, TAU,
                                     KAPPA, THETA, SIGMA_V, RHO, R,
                                     M=M, N=N_STEPS, seed=42)
    mc_prices.append(price)
    mc_ses.append(se)
    print(f"  M = {M:>6d}   price = {price:.4f}   SE = {se:.4f}")

mc_prices = np.array(mc_prices)
mc_ses    = np.array(mc_ses)

# ── O(M^{-1/2}) reference line anchored at first point ────────────────────
ref_se = mc_ses[0] * np.sqrt(PATH_COUNTS[0] / PATH_COUNTS)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Monte Carlo Validation: Price Convergence and Error Decay\n"
    r"ATM Call ($K=100$, $\tau=1\mathrm{y}$, $\boldsymbol{\theta}^*$, antithetic variates)",
    fontsize=11
)

# ── Left panel: MC price convergence ─────────────────────────────────────
ax = axes[0]
ax.plot(PATH_COUNTS, mc_prices,
        "o-", color=BLUE, lw=1.8, ms=6,
        label="MC estimate")
ax.fill_between(PATH_COUNTS,
                mc_prices - 1.96 * mc_ses,
                mc_prices + 1.96 * mc_ses,
                alpha=0.20, color=BLUE, label="95% CI")
ax.axhline(SA_PRICE,
           color=RED, lw=1.8, ls="--",
           label=f"Semi-analytical = {SA_PRICE:.4f}")

ax.set_xscale("log")
ax.set_xlabel("Number of Paths $M$")
ax.set_ylabel("European Call Price")
ax.set_title(r"MC Convergence ($K = 100$, $\tau = 1\mathrm{y}$)")
ax.legend(loc="lower right", framealpha=0.9)

# ── Right panel: SE decay (log-log) ──────────────────────────────────────
ax = axes[1]
ax.loglog(PATH_COUNTS, mc_ses,
          "o-", color=BLUE, lw=1.8, ms=6,
          label="Standard error")
ax.loglog(PATH_COUNTS, ref_se,
          "--", color=RED, lw=1.8,
          label=r"$O(M^{-1/2})$ reference")

ax.set_xlabel("Number of Paths $M$")
ax.set_ylabel("Standard Error")
ax.set_title(r"SE Decay Rate (log-log scale)")
ax.legend(framealpha=0.9)

plt.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════

for ext in ("pdf", "png"):
    fpath = os.path.join(SAVE_DIR, f"fig61_mc_convergence.{ext}")
    plt.savefig(fpath, bbox_inches="tight", dpi=300)
    print(f"Saved: {fpath}")

plt.close()
print("Done.")
# %%
