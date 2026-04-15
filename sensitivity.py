import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
import os

# Save path 
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

# Colours 
C1 = '#1a4f8a'; C2 = '#c0392b'; C3 = '#1e8449'; C4 = '#7d3c98'; GREY = '#7f8c8d'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'lines.linewidth': 2.0, 'figure.dpi': 150,
})

# Base parameters 
KAPPA   = 2.0; THETA = 0.04; SIGMA_V = 0.30; RHO = -0.70; V0 = 0.04
r = 0.05; S0 = 100.0; TAU = 1.0     # fix maturity at 1 year

STRIKES   = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
MONEYNESS = np.log(STRIKES / S0)
bs_flat_vol = np.sqrt(THETA)

# Pricing functions 
def heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau):
    xi  = kappa - rho * sigma_v * 1j * u
    d   = np.sqrt(xi**2 + sigma_v**2 * u * (u + 1j))
    g2  = (xi - d) / (xi + d); ed = np.exp(-d * tau); den = 1.0 - g2 * ed
    D   = ((xi - d) / sigma_v**2) * (1.0 - ed) / den
    C   = (r * 1j * u * tau
           + (kappa * theta / sigma_v**2)
           * ((xi - d) * tau - 2.0 * np.log(den / (1.0 - g2))))
    return np.exp(C + D * V0 + 1j * u * np.log(S0))

def heston_call(S0, K, r, tau, V0, kappa, theta, sigma_v, rho):
    def i1(u): return np.real(np.exp(-1j*u*np.log(K))*heston_cf(u-1j,S0,V0,kappa,theta,sigma_v,rho,r,tau)/heston_cf(-1j,S0,V0,kappa,theta,sigma_v,rho,r,tau)/(1j*u))
    def i2(u): return np.real(np.exp(-1j*u*np.log(K))*heston_cf(u,S0,V0,kappa,theta,sigma_v,rho,r,tau)/(1j*u))
    return S0*(0.5+(1/np.pi)*quad(i1,1e-8,200,limit=80)[0])-K*np.exp(-r*tau)*(0.5+(1/np.pi)*quad(i2,1e-8,200,limit=80)[0])

def bs_call(S, K, r, tau, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d1-sigma*np.sqrt(tau))

def bs_vega(S, K, r, tau, sigma):
    return S*norm.pdf((np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau)))*np.sqrt(tau)

def implied_vol(S, K, r, tau, price):
    sigma = 0.3
    for _ in range(200):
        p = bs_call(S, K, r, tau, sigma); v = bs_vega(S, K, r, tau, sigma)
        if abs(v) < 1e-12: break
        sigma -= (p - price) / v; sigma = max(sigma, 1e-7)
        if abs(p - price) < 1e-9: break
    return sigma

def smile(kappa, theta, sigma_v, rho, V0, tau):
    """Compute IV smile across STRIKES for given parameters."""
    ivs = []
    for K in STRIKES:
        p  = heston_call(S0, K, r, tau, V0, kappa, theta, sigma_v, rho)
        ivs.append(implied_vol(S0, K, r, tau, p) * 100)
    return ivs

# Sensitivity configurations
param_configs = [
    (
        r'Correlation $\rho$ (leverage effect)',
        [(-0.9, 'C1'), (-0.7, 'C2'), (-0.5, 'C3'), (-0.3, 'C4'), (0.0, 'GREY')],
        lambda val: dict(kappa=KAPPA, theta=THETA, sigma_v=SIGMA_V, rho=val, V0=V0),
        r'$\rho$',
    ),
    (
        r'Vol-of-vol $\sigma_v$ (smile curvature)',
        [(0.10, 'C1'), (0.20, 'C2'), (0.30, 'C3'), (0.50, 'C4'), (0.70, 'GREY')],
        lambda val: dict(kappa=KAPPA, theta=THETA, sigma_v=val, rho=RHO, V0=V0),
        r'$\sigma_v$',
    ),
    (
        r'Mean-reversion speed $\kappa$ (smile term structure)',
        [(0.5, 'C1'), (1.0, 'C2'), (2.0, 'C3'), (4.0, 'C4'), (8.0, 'GREY')],
        lambda val: dict(kappa=val, theta=THETA, sigma_v=SIGMA_V, rho=RHO, V0=V0),
        r'$\kappa$',
    ),
    (
        r'Initial variance $V_0$ (ATM level)',
        [(0.01, 'C1'), (0.02, 'C2'), (0.04, 'C3'), (0.08, 'C4'), (0.16, 'GREY')],
        lambda val: dict(kappa=KAPPA, theta=THETA, sigma_v=SIGMA_V, rho=RHO, V0=val),
        r'$V_0$',
    ),
]

color_map = {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'GREY': GREY}

# Plot
print("Computing parameter sensitivity...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for ax, (title, val_color_list, param_fn, pname) in \
        zip(axes.flatten(), param_configs):

    for val, col_key in val_color_list:
        kwargs  = param_fn(val)
        iv_vals = smile(**kwargs, tau=TAU)
        ax.plot(MONEYNESS, iv_vals,
                color=color_map[col_key], linewidth=2.0,
                marker='o', markersize=4,
                label=f'{pname}$={val}$')

    ax.axhline(bs_flat_vol * 100, color='black', linestyle=':',
               linewidth=1.2, alpha=0.6, label='BS flat vol')
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Log-Moneyness $\\ln(K/S_0)$')
    ax.set_ylabel('Implied Volatility (\\%)')
    ax.set_title(f'Effect of {title}  ($\\tau = {TAU}$y)')
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Heston Parameter Sensitivity: Effect on the Implied Volatility Smile',
    fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig07_sensitivity.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'fig07_sensitivity.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: fig07_sensitivity.pdf and fig07_sensitivity.png")
