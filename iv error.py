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

# ── Save path ──────────────────────────────────────────────────────────────
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\images Chapter 2 and 3"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'lines.linewidth': 2.0, 'figure.dpi': 150,
})

# ── Parameters ─────────────────────────────────────────────────────────────
KAPPA   = 2.0; THETA = 0.04; SIGMA_V = 0.30; RHO = -0.70; V0 = 0.04
r = 0.05; S0 = 100.0
STRIKES    = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
MATURITIES = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
bs_flat_vol = np.sqrt(THETA)

# ── Pricing functions ──────────────────────────────────────────────────────
def heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau):
    xi = kappa - rho*sigma_v*1j*u; d = np.sqrt(xi**2+sigma_v**2*u*(u+1j))
    g2 = (xi-d)/(xi+d); ed = np.exp(-d*tau); den = 1.0-g2*ed
    D = ((xi-d)/sigma_v**2)*(1.0-ed)/den
    C = r*1j*u*tau+(kappa*theta/sigma_v**2)*((xi-d)*tau-2.0*np.log(den/(1.0-g2)))
    return np.exp(C+D*V0+1j*u*np.log(S0))

def heston_call(S0, K, r, tau, V0, kappa, theta, sigma_v, rho):
    def i1(u): return np.real(np.exp(-1j*u*np.log(K))*heston_cf(u-1j,S0,V0,kappa,theta,sigma_v,rho,r,tau)/heston_cf(-1j,S0,V0,kappa,theta,sigma_v,rho,r,tau)/(1j*u))
    def i2(u): return np.real(np.exp(-1j*u*np.log(K))*heston_cf(u,S0,V0,kappa,theta,sigma_v,rho,r,tau)/(1j*u))
    return S0*(0.5+(1/np.pi)*quad(i1,1e-8,200,limit=80)[0])-K*np.exp(-r*tau)*(0.5+(1/np.pi)*quad(i2,1e-8,200,limit=80)[0])

def bs_call(S,K,r,tau,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d1-sigma*np.sqrt(tau))

def bs_vega(S,K,r,tau,sigma):
    return S*norm.pdf((np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau)))*np.sqrt(tau)

def implied_vol(S,K,r,tau,price):
    sigma=0.3
    for _ in range(200):
        p=bs_call(S,K,r,tau,sigma); v=bs_vega(S,K,r,tau,sigma)
        if abs(v)<1e-12: break
        sigma-=(p-price)/v; sigma=max(sigma,1e-7)
        if abs(p-price)<1e-9: break
    return sigma

# ── Build surfaces ─────────────────────────────────────────────────────────
print("Building surfaces...")
heston_ivs = np.zeros((len(MATURITIES), len(STRIKES)))
for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        p = heston_call(S0, K, r, tau, V0, KAPPA, THETA, SIGMA_V, RHO)
        heston_ivs[i, j] = implied_vol(S0, K, r, tau, p)

# Calibrate
print("Calibrating...")
def loss(params):
    kappa,theta,sigma_v,rho,v0 = params
    if kappa<=0 or theta<=0 or sigma_v<=0 or not(-1<rho<1) or v0<=0: return 1e10
    total=0.0
    for i,tau in enumerate(MATURITIES):
        for j,K in enumerate(STRIKES):
            try:
                p=heston_call(S0,K,r,tau,v0,kappa,theta,sigma_v,rho)
                iv_m=implied_vol(S0,K,r,tau,p)
                w=bs_vega(S0,K,r,tau,heston_ivs[i,j])
                total+=w**2*(iv_m-heston_ivs[i,j])**2
            except: total+=1e4
    return total

x0 = np.array([KAPPA*1.1, THETA*0.95, SIGMA_V*1.08, RHO*0.93, V0*1.05])
bounds = [(0.1,10),(0.005,0.20),(0.05,1.0),(-0.99,-0.05),(0.005,0.20)]
res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
               options={'ftol':1e-14,'gtol':1e-10,'maxiter':2000})
KC, TC, SVC, RC, V0C = res.x

calib_ivs = np.zeros_like(heston_ivs)
for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        p = heston_call(S0, K, r, tau, V0C, KC, TC, SVC, RC)
        calib_ivs[i, j] = implied_vol(S0, K, r, tau, p)

bs_ivs = np.full_like(heston_ivs, bs_flat_vol)

# ── Plot ───────────────────────────────────────────────────────────────────
iv_err_bs     = np.abs(bs_ivs    - heston_ivs) * 100
iv_err_heston = np.abs(calib_ivs - heston_ivs) * 100
vmax = iv_err_bs.max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
strike_labels   = [str(int(k)) for k in STRIKES]
maturity_labels = [str(t) for t in MATURITIES]

for ax, err, cmap, title in zip(
        axes,
        [iv_err_bs, iv_err_heston],
        ['Reds', 'Blues'],
        ['Black-Scholes IV Error (percentage points)',
         'Calibrated Heston IV Error (percentage points)']):

    im = ax.imshow(err, cmap=cmap, aspect='auto', origin='lower', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(STRIKES))); ax.set_xticklabels(strike_labels)
    ax.set_yticks(range(len(MATURITIES))); ax.set_yticklabels(maturity_labels)
    ax.set_xlabel('Strike $K$'); ax.set_ylabel('Maturity $\\tau$ (years)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='|IV error| (pp)')
    for i in range(len(MATURITIES)):
        for j in range(len(STRIKES)):
            val = err[i, j]
            txt_color = 'white' if val > vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold')

plt.suptitle('Implied Volatility Error Across Strike-Maturity Grid',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig04_iv_error.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'fig04_iv_error.png'), bbox_inches='tight', dpi=200)
plt.close()
print("Saved: fig04_iv_error.pdf and fig04_iv_error.png")
print(f"  BS RMSE:     {np.sqrt(np.mean(iv_err_bs**2)):.4f} pp")
print(f"  Heston RMSE: {np.sqrt(np.mean(iv_err_heston**2)):.6f} pp")

