import numpy as np
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ── True parameters θ* ────────────────────────────────────────────────────
KAPPA   = 2.0
THETA   = 0.04
SIGMA_V = 0.30
RHO     = -0.70
V0      = 0.04
R       = 0.05
S0      = 100.0
M       = 50_000   # paths
N       = 200      # time steps per path

# ── Contracts for Table 6.2 ───────────────────────────────────────────────
CONTRACTS = [
    (90,  1.00),
    (100, 0.25),
    (100, 0.50),
    (100, 1.00),
    (100, 2.00),
    (110, 1.00),
]

# ── Semi-analytical price (Albrecher et al. branch-cut-safe) ──────────────
def heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau):
    i   = 1j
    d   = np.sqrt((kappa - i*rho*sigma_v*u)**2 + sigma_v**2*(u**2 + i*u))
    b   = kappa - i*rho*sigma_v*u
    g   = (b - d) / (b + d)
    D   = ((b - d) / sigma_v**2
           * (1 - np.exp(-d*tau)) / (1 - g*np.exp(-d*tau)))
    C   = (i*u*r*tau
           + kappa*theta/sigma_v**2
           * ((b-d)*tau - 2*np.log((1 - g*np.exp(-d*tau))/(1 - g))))
    return np.exp(C + D*V0 + i*u*np.log(S0))

def sa_price(S0, K, r, tau, V0, kappa, theta, sigma_v, rho):
    i = 1j
    def i1(u):
        return np.real(
            np.exp(-i*u*np.log(K))
            * heston_cf(u-i, S0, V0, kappa, theta, sigma_v, rho, r, tau)
            / heston_cf(-i,  S0, V0, kappa, theta, sigma_v, rho, r, tau)
            / (i*u))
    def i2(u):
        return np.real(
            np.exp(-i*u*np.log(K))
            * heston_cf(u, S0, V0, kappa, theta, sigma_v, rho, r, tau)
            / (i*u))
    P1 = 0.5 + quad(i1, 1e-8, 200, limit=200, epsabs=1e-8)[0] / np.pi
    P2 = 0.5 + quad(i2, 1e-8, 200, limit=200, epsabs=1e-8)[0] / np.pi
    return S0*P1 - K*np.exp(-r*tau)*P2

# ── Monte Carlo with antithetic variates ──────────────────────────────────
def mc_antithetic(K, tau, seed=42):
    np.random.seed(seed)
    dt  = tau / N
    Z1  = np.random.randn(M, N)
    Z2  = RHO*Z1 + np.sqrt(1 - RHO**2)*np.random.randn(M, N)

    def simulate(z1, z2):
        V = np.full(M, V0, float)
        S = np.full(M, S0, float)
        for t in range(N):
            Vp  = np.maximum(V, 0.0)
            V   = V + KAPPA*(THETA - Vp)*dt + SIGMA_V*np.sqrt(Vp*dt)*z2[:, t]
            S   = S * np.exp((R - 0.5*Vp)*dt + np.sqrt(Vp*dt)*z1[:, t])
        return S

    ST_orig = simulate( Z1,  Z2)
    ST_anti = simulate(-Z1, -Z2)

    disc   = np.exp(-R*tau)
    payoff = disc * 0.5*(np.maximum(ST_orig - K, 0.0)
                       + np.maximum(ST_anti - K, 0.0))
    price  = payoff.mean()
    se     = payoff.std(ddof=1) / np.sqrt(M)
    return price, se

# ── Run ────────────────────────────────────────────────────────────────────
print("=" * 72)
print("TABLE 6.2: Monte Carlo Validation Against Semi-Analytical Prices")
print(f"S0 = {S0},  M = {M:,} paths (antithetic),  N = {N} time steps")
print("=" * 72)
print()
print(f"{'K':>5} {'tau':>6} {'SA Price':>10} {'MC Price':>10} "
      f"{'Std Error':>11} {'|Error|':>9} {'Within 2SE':>11}")
print("-" * 72)

all_within = True
for K, tau in CONTRACTS:
    p_sa         = sa_price(S0, K, R, tau, V0, KAPPA, THETA, SIGMA_V, RHO)
    p_mc, se     = mc_antithetic(K, tau)
    error        = abs(p_sa - p_mc)
    within_2se   = error < 2*se
    if not within_2se:
        all_within = False
    flag = "YES" if within_2se else "*** NO ***"
    print(f"{K:>5} {tau:>6.2f} {p_sa:>10.4f} {p_mc:>10.4f} "
          f"{se:>11.4f} {error:>9.4f} {flag:>11}")

print()
print("=" * 72)
if all_within:
    print("All MC prices lie within 2 standard errors of the SA benchmark.")
else:
    print("WARNING: Some MC prices lie outside 2 SE — check implementation.")
print()
# %%
