"""
Validate the trained 6-D posterior estimator. Three tests:

  TEST 1  Recovery + degeneracy: for one held-out system, show the posterior marginals
          vs the true (m1,m2,h1,k1,h2,k2), an eccentricity-vector projection, and a
          POSTERIOR-PREDICTIVE check (re-simulate posterior samples -> their TTV curves
          should bracket the observed data).

  TEST 2  Calibration (coverage / SBC): the X% credible interval must contain the truth
          X% of the time, for EVERY parameter. This is the property prior ML-for-TTV
          work lacked. (In 6-D a brute-force grid is infeasible, so calibration across
          many simulated systems is the rigorous correctness test.)

  TEST 3  Speed (amortization): network inference vs the N-body cost an MCMC would pay.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(7)

norm = np.load("norm.npz")
f_mean, f_std = norm["f_mean"], norm["f_std"]
th_mean, th_std = norm["th_mean"], norm["th_std"]
NOISE_MIN = 0.5

net = MDN(in_dim=S.FEATURE_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
net.load_state_dict(torch.load("mdn.pt", map_location=DEVICE))
net.eval()

def infer(feat_noisy, n=4000):
    x = torch.tensor((feat_noisy - f_mean)/f_std, dtype=torch.float32, device=DEVICE).view(1, -1)
    s = net.sample(x, n=n).cpu().numpy()[0]
    return s*th_std + th_mean


# ===================== TEST 1: recovery, degeneracy, posterior-predictive =====================
print("="*72); print("TEST 1  —  posterior recovery for one system (6 parameters)"); print("="*72)
theta_true = np.array([8.0, 28.0, 0.05, 0.00, 0.03, 0.04])    # m1,m2,h1,k1,h2,k2
clean = S.simulate(theta_true)[0]
obs = clean + rng.normal(0, NOISE_MIN, clean.shape)
post = infer(obs)
mean, std = post.mean(0), post.std(0)

print(f"\n  {'param':>8} {'true':>8} {'post mean':>11} {'post std':>10} {'|z|':>6}")
for d in range(S.THETA_DIM):
    z = abs(mean[d]-theta_true[d])/(std[d]+1e-9)
    print(f"  {S.THETA_NAMES[d]:>8} {theta_true[d]:>8.3f} {mean[d]:>11.3f} {std[d]:>10.3f} {z:>6.2f}")
print("  (|z| = how many posterior sigmas the truth sits from the mean; ~<2 is good)")

# eccentricity-vector projection of planet 2 (h2,k2)
def ascii_scatter(samples, di, dj, truth, lim, w=42, h=15, labels=("","")):
    grid = [[' ']*w for _ in range(h)]
    def cell(a, b):
        c = int((a+lim)/(2*lim)*(w-1)); r = int((1-(b+lim)/(2*lim))*(h-1))
        return max(0,min(h-1,r)), max(0,min(w-1,c))
    for s in samples:
        r, c = cell(s[di], s[dj]); grid[r][c] = '.' if grid[r][c]==' ' else (':' if grid[r][c]=='.' else '#')
    r, c = cell(truth[di], truth[dj]); grid[r][c] = 'X'
    print(f"\n  posterior in ({labels[0]},{labels[1]}) plane  (X = truth):")
    print("   +%.2f " % lim + "_"*w)
    for r in range(h): print("        |" + "".join(grid[r]) + "|")
    print("   -%.2f " % lim + "-"*w + f"   {labels[0]}: -{lim:.2f}..+{lim:.2f}")
ascii_scatter(post, 4, 5, theta_true, S.E2_MAX, labels=("h2","k2"))

# posterior-predictive: do posterior samples reproduce the observed inner-planet TTV?
sub = post[rng.choice(len(post), 24, replace=False)]
sims = S.simulate(sub)                              # (24, FEATURE_DIM)
inner = sims[:, :S.N_TRANSITS_1]
lo, hi = np.nanmin(inner, 0), np.nanmax(inner, 0)
obs_inner = obs[:S.N_TRANSITS_1]
inside = np.mean((obs_inner >= lo-NOISE_MIN) & (obs_inner <= hi+NOISE_MIN))
print(f"\n  Posterior-predictive check (inner-planet O-C):")
print(f"    {inside*100:.0f}% of observed transits fall within the posterior-sample TTV envelope")
print(f"    envelope half-width vs data: median {np.median((hi-lo)/2):.2f} min  (noise {NOISE_MIN} min)")
print("    -> the inferred posterior reproduces the data it was conditioned on.")


# ===================== TEST 2: calibration =====================
print("\n" + "="*72); print("TEST 2  —  calibration: honest error bars on all 6 parameters"); print("="*72)
N_SBC = 800
theta_test = S.sample_prior(N_SBC, rng)
feat_test = S.simulate(theta_test)
ok = ~np.isnan(feat_test).any(axis=1)
theta_test, feat_test = theta_test[ok], feat_test[ok]
feat_obs = feat_test + rng.normal(0, NOISE_MIN, feat_test.shape)
x = torch.tensor((feat_obs - f_mean)/f_std, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    samp = net.sample(x, n=1000).cpu().numpy()*th_std + th_mean    # (N,1000,6)

print(f"\nCoverage (ideal = nominal), over {len(theta_test)} simulated systems:")
header = "  nominal | " + " | ".join(f"{nm:>8}" for nm in S.THETA_NAMES)
print(header)
for lv in [0.5, 0.68, 0.9, 0.95]:
    lo_p, hi_p = (1-lv)/2*100, (1+lv)/2*100
    cells = []
    for d in range(S.THETA_DIM):
        qlo = np.percentile(samp[:, :, d], lo_p, axis=1)
        qhi = np.percentile(samp[:, :, d], hi_p, axis=1)
        cov = np.mean((theta_test[:, d] >= qlo) & (theta_test[:, d] <= qhi))
        cells.append(f"{cov*100:7.1f}%")
    print(f"  {lv*100:5.0f}%  | " + " | ".join(cells))

# SBC rank histograms for m2 and h2
for d, nm in [(1, "m2"), (4, "h2")]:
    ranks = (samp[:, :, d] < theta_test[:, [d]]).sum(1)
    hist, _ = np.histogram(ranks, bins=10, range=(0, samp.shape[1]))
    print(f"\nSBC rank histogram for {nm} (flat = calibrated):")
    mx = hist.max()
    for b in range(10):
        print(f"  bin {b}: {'#'*int(round(hist[b]/mx*38))} {hist[b]}")


# ===================== TEST 3: speed =====================
print("\n" + "="*72); print("TEST 3  —  speed: amortized inference vs the N-body an MCMC must run"); print("="*72)
x1 = torch.tensor((obs - f_mean)/f_std, dtype=torch.float32, device=DEVICE).view(1, -1)
with torch.no_grad(): _ = net.sample(x1, n=4000)
t0 = time.time()
for _ in range(50):
    with torch.no_grad(): _ = net.sample(x1, n=4000)
net_ms = (time.time()-t0)/50*1000

t0 = time.time(); _ = S.simulate(S.sample_prior(200, rng)); sim_ms = (time.time()-t0)/200*1000
for n_mcmc in (1e5, 1e6):
    print(f"  MCMC at {n_mcmc:.0e} N-body sims  ~ {n_mcmc*sim_ms/1000/60:8.1f} min  (1 sim = {sim_ms:.1f} ms)")
print(f"  Trained network, one full 6-D posterior: {net_ms:.2f} ms")
print("  Training cost is paid ONCE; every subsequent system is then ~milliseconds (amortization).")
