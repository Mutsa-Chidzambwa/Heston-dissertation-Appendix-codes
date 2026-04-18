import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings("ignore")

# Output directory
import os
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\Python codes_Ch6"
os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE — MATLAB-style white background, jet colormap, fine mesh
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 10,
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10.5,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'figure.facecolor': 'white',
    'axes.facecolor'  : 'white',
    'figure.dpi'      : 150,
})

def style_matlab(ax):
    ax.set_facecolor('white')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor('#f5f5f5')
        pane.set_alpha(1.0)
        pane.set_edgecolor('#aaaaaa')
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid'].update({'color': '#bbbbbb', 'linestyle': ':', 'linewidth': 0.6})
    ax.tick_params(colors='black', labelsize=8.5)
    for attr in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label, ax.title]:
        attr.set_color('black')

def add_colorbar(fig, surf, ax, label, shrink=0.50):
    cb = fig.colorbar(surf, ax=ax, shrink=shrink, pad=0.08, fraction=0.025)
    cb.set_label(label, fontsize=9)
    return cb

# ══════════════════════════════════════════════════════════════════════════════
#  HESTON PRICING ENGINE  (Albrecher et al. branch-cut-safe formulation)
# ══════════════════════════════════════════════════════════════════════════════
S0 = 100.; r = 0.; V0 = 0.04
kappa = 2.; theta = 0.04; sigma = 0.30; rho = -0.70

STRIKES    = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
MATURITIES = np.array([0.25, 0.50, 1.00, 1.50, 2.00])
LOG_MONEY  = np.log(STRIKES / S0)

def heston_cf(u, S, v, tau, kappa, theta, sigma, rho, r):
    b = kappa - 1j*rho*sigma*u
    d = np.sqrt(b**2 + sigma**2*(1j*u + u**2))
    g = (b - d) / (b + d)
    D = ((b - d)/sigma**2) * ((1 - np.exp(-d*tau)) / (1 - g*np.exp(-d*tau)))
    C = (kappa*theta/sigma**2) * ((b-d)*tau - 2*np.log((1 - g*np.exp(-d*tau))/(1 - g))) \
        + 1j*u*r*tau
    return np.exp(C + D*v + 1j*u*np.log(S))

def heston_call(S, v, K, tau, kappa, theta, sigma, rho, r):
    cf0 = heston_cf(-1j, S, v, tau, kappa, theta, sigma, rho, r)
    I1  = lambda u: np.real(np.exp(-1j*u*np.log(K)) *
                            heston_cf(u-1j, S, v, tau, kappa, theta, sigma, rho, r) /
                            (1j*u*cf0))
    I2  = lambda u: np.real(np.exp(-1j*u*np.log(K)) *
                            heston_cf(u, S, v, tau, kappa, theta, sigma, rho, r) /
                            (1j*u))
    P1 = 0.5 + (1/np.pi)*quad(I1, 1e-8, 200, limit=200)[0]
    P2 = 0.5 + (1/np.pi)*quad(I2, 1e-8, 200, limit=200)[0]
    return S*P1 - K*np.exp(-r*tau)*P2

def bs_call(S, K, tau, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*tau) / (sig*np.sqrt(tau))
    d2 = d1 - sig*np.sqrt(tau)
    return S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

def implied_vol(price, S, K, tau, r):
    try:
        if bs_call(S,K,tau,r,10.) < price or bs_call(S,K,tau,r,1e-6) > price:
            return np.nan
        return brentq(lambda s: bs_call(S,K,tau,r,s) - price, 1e-6, 10., xtol=1e-10)
    except:
        return np.nan

def vega_bs(S, K, tau, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*tau) / (sig*np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)

def smooth(Z, xs, ys, xs_fine, ys_fine):
    spl = RectBivariateSpline(ys, xs, Z, kx=3, ky=3)
    return spl(ys_fine, xs_fine)

# Build raw surfaces
print("Building IV and price surfaces...")
IV_H  = np.zeros((len(MATURITIES), len(STRIKES)))
PH    = np.zeros_like(IV_H)
PBS   = np.zeros_like(IV_H)

for i, tau in enumerate(MATURITIES):
    for j, K in enumerate(STRIKES):
        p  = heston_call(S0, V0, K, tau, kappa, theta, sigma, rho, r)
        iv = implied_vol(p, S0, K, tau, r)
        IV_H[i,j] = iv * 100
        PH[i,j]   = p
        PBS[i,j]  = bs_call(S0, K, tau, r, 0.20)

IV_ERR = np.abs(IV_H - 20.)
PDIFF  = PH - PBS

# Smooth to dense grids
lm_fine  = np.linspace(LOG_MONEY[0],  LOG_MONEY[-1],  40)
tau_fine = np.linspace(MATURITIES[0], MATURITIES[-1], 40)
K_fine   = np.linspace(STRIKES[0],    STRIKES[-1],    40)

IV_s   = smooth(IV_H,   LOG_MONEY, MATURITIES, lm_fine,  tau_fine)
ERR_s  = smooth(IV_ERR, LOG_MONEY, MATURITIES, lm_fine,  tau_fine)
PH_s   = smooth(PH,     STRIKES,   MATURITIES, K_fine,   tau_fine)
PBS_s  = smooth(PBS,    STRIKES,   MATURITIES, K_fine,   tau_fine)
DIFF_s = smooth(PDIFF,  STRIKES,   MATURITIES, K_fine,   tau_fine)

KK,  TT  = np.meshgrid(lm_fine,  tau_fine)
KK2, TT2 = np.meshgrid(K_fine,   tau_fine)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE A — IV surface (left) + IV error surface (right)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 6.2), facecolor='white')
fig.suptitle('3D Implied Volatility Surface Analysis',
             fontsize=13, fontweight='bold', color='black', y=1.01)

# Left: Heston IV surface
ax1 = fig.add_subplot(121, projection='3d')
style_matlab(ax1)

s1 = ax1.plot_surface(KK, TT, IV_s, cmap='jet', alpha=1.0,
                      rstride=1, cstride=1, linewidth=0.25,
                      edgecolor='k', antialiased=True)

# BS flat reference plane
BS_Z = np.full_like(KK, 20.)
ax1.plot_surface(KK, TT, BS_Z, alpha=0.28, color='royalblue',
                 linewidth=0.5, edgecolor='royalblue')
for lm in [lm_fine[0], lm_fine[-1]]:
    ax1.plot([lm,lm], [tau_fine[0],tau_fine[-1]], [20,20],
             'royalblue', lw=1.6, alpha=0.9)
for t in [tau_fine[0], tau_fine[-1]]:
    ax1.plot([lm_fine[0],lm_fine[-1]], [t,t], [20,20],
             'royalblue', lw=1.6, alpha=0.9)

# ATM spine
ax1.plot(np.zeros(len(tau_fine)), tau_fine,
         np.interp(tau_fine, MATURITIES, IV_H[:,4]),
         'w-', lw=1.8, zorder=6)

ax1.set_xlabel('Log-moneyness  $\\ln(K/S_0)$', labelpad=8)
ax1.set_ylabel('Maturity  $\\tau$ (years)',     labelpad=8)
ax1.set_zlabel('Implied Volatility (%)',         labelpad=8)
ax1.set_title('Heston Implied Volatility Surface\n'
              '(blue plane = Black--Scholes flat 20\\%)')
add_colorbar(fig, s1, ax1, 'IV (%)')
ax1.view_init(elev=28, azim=-50)

# Right: BS IV error surface
ax2 = fig.add_subplot(122, projection='3d')
style_matlab(ax2)

s2 = ax2.plot_surface(KK, TT, ERR_s, cmap='jet', alpha=1.0,
                      rstride=1, cstride=1, linewidth=0.25,
                      edgecolor='k', antialiased=True)

ax2.set_xlabel('Log-moneyness  $\\ln(K/S_0)$', labelpad=8)
ax2.set_ylabel('Maturity  $\\tau$ (years)',     labelpad=8)
ax2.set_zlabel('$|$IV Error$|$ (pp)',            labelpad=8)
ax2.set_title('Black--Scholes IV Error Surface\n'
              r'$|\sigma_{\mathrm{IV}}^{\mathrm{BS}}-\sigma_{\mathrm{IV}}^{\mathrm{Heston}}|$')
add_colorbar(fig, s2, ax2, '|IV Error| (pp)')
ax2.view_init(elev=28, azim=-50)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_A_iv_surface.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_A_iv_surface.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Figure A")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE B — call price surface (left) + pricing difference (right)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 6.2), facecolor='white')
fig.suptitle('3D European Call Price Surface Comparison',
             fontsize=13, fontweight='bold', color='black', y=1.01)

# ── Left: overlaid Heston + BS ────────────────────────────────────────────────
ax1 = fig.add_subplot(121, projection='3d')
style_matlab(ax1)

ax1.plot_surface(KK2, TT2, PBS_s, alpha=0.55, color='#e05c5c',
                 linewidth=0.4, edgecolor='#c03030')
s_h = ax1.plot_surface(KK2, TT2, PH_s, cmap='jet', alpha=0.88,
                       rstride=1, cstride=1, linewidth=0.25,
                       edgecolor='k', antialiased=True)

legend_els = [Patch(facecolor='steelblue', alpha=0.9, label='Heston'),
              Patch(facecolor='#e05c5c',   alpha=0.7, label='Black--Scholes')]
ax1.legend(handles=legend_els, fontsize=9, loc='upper right',
           facecolor='white', edgecolor='#aaaaaa')

ax1.set_xlabel('Strike  $K$',            labelpad=8)
ax1.set_ylabel('Maturity  $\\tau$ (yr)', labelpad=8)
ax1.set_zlabel('Call Price',             labelpad=8)
ax1.set_title('Call Price Surfaces: Heston vs Black--Scholes')
add_colorbar(fig, s_h, ax1, 'Heston Price')
ax1.view_init(elev=25, azim=-55)

# Right: pricing difference
ax2 = fig.add_subplot(122, projection='3d')
style_matlab(ax2)

vcen = mcolors.TwoSlopeNorm(vmin=DIFF_s.min(), vcenter=0, vmax=DIFF_s.max())
s2 = ax2.plot_surface(KK2, TT2, DIFF_s, cmap='RdBu_r', norm=vcen,
                      alpha=1.0, rstride=1, cstride=1,
                      linewidth=0.25, edgecolor='k', antialiased=True)
ax2.plot_surface(KK2, TT2, np.zeros_like(DIFF_s), alpha=0.20,
                 color='grey', linewidth=0)
for sk in [K_fine[0], K_fine[-1]]:
    ax2.plot([sk,sk], [tau_fine[0],tau_fine[-1]], [0,0], 'grey', lw=1.0, alpha=0.5)
for t in [tau_fine[0], tau_fine[-1]]:
    ax2.plot([K_fine[0],K_fine[-1]], [t,t], [0,0], 'grey', lw=1.0, alpha=0.5)

ax2.set_xlabel('Strike  $K$',            labelpad=8)
ax2.set_ylabel('Maturity  $\\tau$ (yr)', labelpad=8)
ax2.set_zlabel('Price Difference',       labelpad=8)
ax2.set_title('Pricing Difference: Heston $-$ Black-Scholes\n'
              r'(red = Heston higher; blue = Heston lower)')

add_colorbar(fig, s2, ax2, 'Heston $-$ BS')
ax2.view_init(elev=25, azim=-55)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_B_price_surface.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_B_price_surface.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Figure B")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE C — calibration loss surface  (toned-down: coolwarm, not jet)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing calibration loss surface...")
mkt_iv_frac = IV_H / 100.
kappa_grid  = np.linspace(0.5, 5.0, 32)
rho_grid    = np.linspace(-0.95, -0.10, 32)
KAP, RHO    = np.meshgrid(kappa_grid, rho_grid)
LOSS        = np.zeros_like(KAP)

for ii in range(KAP.shape[0]):
    for jj in range(KAP.shape[1]):
        loss = 0.
        for i, tau in enumerate(MATURITIES):
            for j, K in enumerate(STRIKES):
                p   = heston_call(S0, V0, K, tau, KAP[ii,jj], theta, sigma, RHO[ii,jj], r)
                ivm = implied_vol(p, S0, K, tau, r)
                if np.isnan(ivm): continue
                w    = vega_bs(S0, K, tau, r, mkt_iv_frac[i,j])
                loss += w**2 * (ivm - mkt_iv_frac[i,j])**2
        LOSS[ii,jj] = np.log1p(loss * 1e6)

# smooth
kg_f   = np.linspace(kappa_grid[0], kappa_grid[-1], 60)
rg_f   = np.linspace(rho_grid[0],   rho_grid[-1],   60)
LOSS_s = smooth(LOSS, kappa_grid, rho_grid, kg_f, rg_f)
KAP_f, RHO_f = np.meshgrid(kg_f, rg_f)

mi      = np.unravel_index(LOSS_s.argmin(), LOSS_s.shape)
km      = KAP_f[mi]; rm = RHO_f[mi]; lm_min = LOSS_s[mi]
floor_z = lm_min - 1.2

spike_col = '#1a1a6e'

# Figure C: use GridSpec to guarantee both subplots render
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(15, 6.2), facecolor='white')
fig.suptitle('3D Calibration Loss Surface  ($\\kappa$--$\\rho$ Slice)',
             fontsize=13, fontweight='bold', color='black', y=1.01)

gs = GridSpec(1, 2, figure=fig, wspace=0.35)

# Left: 3D loss surface
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
style_matlab(ax1)

surf = ax1.plot_surface(KAP_f, RHO_f, LOSS_s,
                        cmap='coolwarm_r', alpha=1.0,
                        rstride=1, cstride=1,
                        linewidth=0.20, edgecolor='k', antialiased=True)

ax1.plot([km, km], [rm, rm], [floor_z, lm_min],
         color=spike_col, lw=2.5, zorder=15)
ax1.scatter([km], [rm], [lm_min],
            color=spike_col, s=90, zorder=16,
            edgecolors='white', linewidths=1.2)
ax1.scatter([km], [rm], [floor_z],
            color=spike_col, s=35, zorder=16, alpha=0.7)
ax1.plot([kg_f[0], km], [rm, rm], [floor_z, floor_z],
         color=spike_col, lw=1.0, alpha=0.5, ls='--')
ax1.plot([km, km], [rg_f[0], rm], [floor_z, floor_z],
         color=spike_col, lw=1.0, alpha=0.5, ls='--')

ax1.set_xlabel('Mean-reversion  $\\kappa$',  labelpad=8)
ax1.set_ylabel('Correlation  $\\rho$',        labelpad=8)
ax1.set_zlabel('$\\log(1+L{\\times}10^6)$',  labelpad=8)
ax1.set_title('Vega-Weighted Calibration Loss\n'
              r'($\sigma_v,\theta,V_0$ fixed at $\theta^*$; $\kappa$--$\rho$ slice)')

leg_el = [Line2D([0],[0], marker='o', color='w',
                 markerfacecolor=spike_col, markersize=9,
                 label=r'Global minimum  $\theta^*$')]
ax1.legend(handles=leg_el, fontsize=9, loc='upper right',
           facecolor='white', edgecolor='#aaaaaa')
add_colorbar(fig, surf, ax1, 'Loss (log scale)')
ax1.view_init(elev=30, azim=40)

# Right: contour top-down
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')
for sp in ax2.spines.values():
    sp.set_edgecolor('#aaaaaa')

cnt = ax2.contourf(KAP_f, RHO_f, LOSS_s, levels=30, cmap='coolwarm_r')
ax2.contour(KAP_f, RHO_f, LOSS_s, levels=18,
            colors='white', linewidths=0.5, alpha=0.40)
ax2.scatter([km], [rm], color=spike_col, s=130, zorder=10,
            edgecolors='white', linewidths=1.0,
            label=f'$\\kappa^*=2.0,\\ \\rho^*=-0.70$')
ax2.set_xlabel('Mean-reversion  $\\kappa$')
ax2.set_ylabel('Correlation  $\\rho$')
ax2.set_title('Top-Down View: Loss Contours\n(global minimum marked)')
ax2.legend(fontsize=9, loc='upper right',
           facecolor='white', edgecolor='#aaaaaa')
cb2 = fig.colorbar(cnt, ax=ax2)
cb2.set_label('Loss (log scale)', fontsize=9)

import os
SAVE_DIR = r"C:\Users\User\OneDrive\Desktop\HFM470 Dissertation\Python codes_Ch6"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_C_loss_surface.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(SAVE_DIR, 'fig_3d_C_loss_surface.pdf'),
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Figure C — all done!")