import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
import os

# ── Save path ──────────────────────────────────────────────────────────────
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────
C1 = '#1a4f8a'; C2 = '#c0392b'; C3 = '#1e8449'; C4 = '#7d3c98'; GREY = '#7f8c8d'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'lines.linewidth': 2.0, 'figure.dpi': 150,
})

# ── Parameters ─────────────────────────────────────────────────────────────
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

# ── Pricing functions ──────────────────────────────────────────────────────
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

# ── Compute surfaces ───────────────────────────────────────────────────────
print("Computing surfaces...")
heston_prices = np.zeros((len(MATURITIES), len(STRIKES)))
heston_ivs    = np.zeros_like(heston_prices)
bs_prices     = np.zeros_like(heston_prices)

for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        hp = heston_call(S0, K, r, tau, V0, KAPPA, THETA, SIGMA_V, RHO)
        heston_prices[i, j] = hp
        heston_ivs[i, j]    = implied_vol(S0, K, r, tau, hp)
        bs_prices[i, j]     = bs_call(S0, K, r, tau, bs_flat_vol)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sel_idx    = [0, 2, 4]           # tau = 0.25, 1.0, 2.0
sel_colors = [C1, C3, C4]

# Left panel: call prices
ax = axes[0]
for idx, i in enumerate(sel_idx):
    tau = MATURITIES[i]
    ax.plot(STRIKES, heston_prices[i], color=sel_colors[idx],
            linewidth=2.2, marker='o', markersize=6,
            label=f'Heston  $\\tau={tau}$y')
    ax.plot(STRIKES, bs_prices[i], color=sel_colors[idx],
            linewidth=1.8, linestyle='--', marker='^', markersize=5,
            alpha=0.75, label=f'BS  $\\tau={tau}$y')

ax.set_xlabel('Strike $K$')
ax.set_ylabel('European Call Price')
ax.set_title('Heston vs Black-Scholes: Call Prices')
ax.legend(ncol=2, fontsize=9, framealpha=0.92)
ax.grid(True, alpha=0.3)

# Right panel: implied volatility
ax = axes[1]
for idx, i in enumerate(sel_idx):
    tau = MATURITIES[i]
    ax.plot(MONEYNESS, heston_ivs[i] * 100, color=sel_colors[idx],
            linewidth=2.2, marker='o', markersize=6,
            label=f'Heston  $\\tau={tau}$y')

ax.axhline(bs_flat_vol * 100, color=C2, linestyle='--', linewidth=2.0,
           label=f'Black-Scholes $\\sigma = {bs_flat_vol*100:.0f}\\%$')
ax.axvline(0, color='grey', linestyle=':', linewidth=1.0, alpha=0.7)
ax.set_xlabel('Log-Moneyness  $\\ln(K/S_0)$')
ax.set_ylabel('Implied Volatility (\\%)')
ax.set_title('Heston vs Black-Scholes: Implied Volatility')
ax.legend(ncol=2, fontsize=9, framealpha=0.92)
ax.grid(True, alpha=0.3)

plt.suptitle('Heston Model vs Black-Scholes: Prices and Implied Volatility',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig02_heston_vs_bs.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'fig02_heston_vs_bs.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: fig02_heston_vs_bs.pdf and fig02_heston_vs_bs.png")







