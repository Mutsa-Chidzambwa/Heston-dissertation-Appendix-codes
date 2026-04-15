import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import os

# Save path
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

#  Colours 
C1 = '#1a4f8a'; C2 = '#c0392b'; C3 = '#1e8449'; C4 = '#7d3c98'; GREY = '#7f8c8d'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'lines.linewidth': 2.0, 'figure.dpi': 150,
})

#  Parameters 
KAPPA   = 2.0
THETA   = 0.04
SIGMA_V = 0.30
RHO     = -0.70
V0      = 0.04
r       = 0.05
S0      = 100.0

STRIKES    = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
MATURITIES = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
MONEYNESS  = np.log(STRIKES / S0)
bs_flat_vol = np.sqrt(THETA)

# Pricing functions
def heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau):
    xi  = kappa - rho * sigma_v * 1j * u
    d   = np.sqrt(xi**2 + sigma_v**2 * u * (u + 1j))
    g2  = (xi - d) / (xi + d)
    ed  = np.exp(-d * tau)
    den = 1.0 - g2 * ed
    D   = ((xi - d) / sigma_v**2) * (1.0 - ed) / den
    C   = (r * 1j * u * tau
           + (kappa * theta / sigma_v**2)
           * ((xi - d) * tau - 2.0 * np.log(den / (1.0 - g2))))
    return np.exp(C + D * V0 + 1j * u * np.log(S0))

def heston_call(S0, K, r, tau, V0, kappa, theta, sigma_v, rho):
    def i1(u):
        return np.real(np.exp(-1j * u * np.log(K))
                       * heston_cf(u - 1j, S0, V0, kappa, theta, sigma_v, rho, r, tau)
                       / heston_cf(-1j,    S0, V0, kappa, theta, sigma_v, rho, r, tau)
                       / (1j * u))
    def i2(u):
        return np.real(np.exp(-1j * u * np.log(K))
                       * heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau)
                       / (1j * u))
    P1 = 0.5 + (1.0 / np.pi) * quad(i1, 1e-8, 200, limit=80)[0]
    P2 = 0.5 + (1.0 / np.pi) * quad(i2, 1e-8, 200, limit=80)[0]
    return S0 * P1 - K * np.exp(-r * tau) * P2

def bs_call(S, K, r, tau, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_vega(S, K, r, tau, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)

def implied_vol(S, K, r, tau, price):
    sigma = 0.3
    for _ in range(200):
        p = bs_call(S, K, r, tau, sigma)
        v = bs_vega(S, K, r, tau, sigma)
        if abs(v) < 1e-12: break
        sigma -= (p - price) / v
        sigma  = max(sigma, 1e-7)
        if abs(p - price) < 1e-9: break
    return sigma

# Build market surface 
print("Building market surface...")
heston_ivs = np.zeros((len(MATURITIES), len(STRIKES)))
for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        p = heston_call(S0, K, r, tau, V0, KAPPA, THETA, SIGMA_V, RHO)
        heston_ivs[i, j] = implied_vol(S0, K, r, tau, p)

# Calibrate
print("Running calibration...")

def loss(params):
    kappa, theta, sigma_v, rho, v0 = params
    if kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-1 < rho < 1) or v0 <= 0:
        return 1e10
    total = 0.0
    for i, tau in enumerate(MATURITIES):
        for j, K in enumerate(STRIKES):
            try:
                p      = heston_call(S0, K, r, tau, v0, kappa, theta, sigma_v, rho)
                iv_mod = implied_vol(S0, K, r, tau, p)
                w      = bs_vega(S0, K, r, tau, heston_ivs[i, j])
                total += w**2 * (iv_mod - heston_ivs[i, j])**2
            except Exception:
                total += 1e4
    return total

x0     = np.array([KAPPA * 1.1, THETA * 0.95, SIGMA_V * 1.08, RHO * 0.93, V0 * 1.05])
bounds = [(0.1, 10), (0.005, 0.20), (0.05, 1.0), (-0.99, -0.05), (0.005, 0.20)]
result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
                  options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000})
KAPPA_C, THETA_C, SIGMA_V_C, RHO_C, V0_C = result.x
print(f"  Calibrated: kappa={KAPPA_C:.4f}, theta={THETA_C:.4f}, "
      f"sigma_v={SIGMA_V_C:.4f}, rho={RHO_C:.4f}, V0={V0_C:.4f}")

calib_ivs = np.zeros_like(heston_ivs)
for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        p = heston_call(S0, K, r, tau, V0_C, KAPPA_C, THETA_C, SIGMA_V_C, RHO_C)
        calib_ivs[i, j] = implied_vol(S0, K, r, tau, p)
# Plot
# Plot 
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, i in enumerate([0, 2, 4]):
    tau = MATURITIES[i]
    ax  = axes[idx]

    # Plot BS flat line first (bottom layer)
    ax.axhline(bs_flat_vol * 100, color=C2, linestyle=':', linewidth=1.8,
               label=f'Black-Scholes $\\sigma={bs_flat_vol*100:.0f}\\%$',
               zorder=1)

    # Plot calibrated Heston second (middle layer)
    ax.plot(MONEYNESS, calib_ivs[i] * 100,
            color=C3, linewidth=3.5, linestyle='--',
            zorder=2, label='Calibrated Heston')

    # Plot market on top (top layer) — larger markers, thinner line
    # so the markers stand out above the green line
    ax.plot(MONEYNESS, heston_ivs[i] * 100,
            color=C1, linewidth=0,
            marker='o', markersize=9, markeredgewidth=1.5,
            markerfacecolor=C1, markeredgecolor='white',
            zorder=3, label='Market (simulated Heston)')

    ax.set_xlabel('Log-Moneyness $\\ln(K/S_0)$')
    ax.set_ylabel('Implied Vol (\\%)')
    ax.set_title(f'$\\tau = {tau}$ year{"s" if tau > 1 else ""}')
    ax.legend(fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.3)

plt.suptitle('Calibration Fit: Market vs Calibrated Heston vs Black-Scholes',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig03_calib_fit.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'fig03_calib_fit.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: fig03_calib_fit.pdf and fig03_calib_fit.png")
