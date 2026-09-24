"""
Hybrid in the near-resonance regime, stage 3 (torch env): evaluate the cross-resonance hybrid MDN.
Two things:
  (1) The near-resonance signature: m2 mass posterior width (posterior/prior) vs period ratio,
      should PEAK at the 2:1 separatrix (ratio 2.0), the informativeness collapse / degeneracy
      that our REBOUND cross_resonance run found, now reproduced in jaxttv physics.
  (2) Calibration across the resonance (coverage), the posterior must stay honest even as it widens.
Writes jaxttv_resonance.png + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import MDN

NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]
PRIOR_STD = np.array([(15-3)/np.sqrt(12), (45-8)/np.sqrt(12), 0.075, 0.075, 0.075, 0.075])
NOISE_MIN = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]

d = np.load("jaxttv_res_test.npz")
theta, ratio, feats = d["theta"], d["ratio"], d["feats"]
norm = np.load("norm_jaxttv_res.npz")
f_mean, f_std, th_mean, th_std = norm["f_mean"], norm["f_std"], norm["th_mean"], norm["th_std"]

rng = np.random.default_rng(7)
obs = feats + rng.normal(0, NOISE_MIN, feats.shape)
X = np.concatenate([obs, ratio[:, None]], axis=1)
x = torch.tensor((X - f_mean) / f_std, dtype=torch.float32, device=DEVICE)
net = MDN(in_dim=X.shape[1], theta_dim=6).to(DEVICE)
net.load_state_dict(torch.load("mdn_jaxttv_res.pt", map_location=DEVICE)); net.eval()
samp = np.empty((len(theta), 1000, 6), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(x), 1000):
        samp[i:i+1000] = net.sample(x[i:i+1000], n=1000).cpu().numpy() * th_std + th_mean

# (1) m2 posterior width vs period ratio
grid = np.unique(ratio)
m2w, m2w_err = [], []
for r in grid:
    m = ratio == r
    w = samp[m][:, :, 1].std(1) / PRIOR_STD[1]      # per-system m2 posterior/prior width
    m2w.append(w.mean()); m2w_err.append(w.std() / np.sqrt(len(w)))
m2w, m2w_err = np.array(m2w), np.array(m2w_err)
sep = np.argmin(np.abs(grid - 2.0))
off = np.argmin(np.abs(grid - 1.90))
print(f"m2 posterior/prior width: at separatrix (r={grid[sep]:.3f}) {m2w[sep]:.3f}  vs  "
      f"off-res (r={grid[off]:.3f}) {m2w[off]:.3f}  -> collapse factor {m2w[sep]/m2w[off]:.2f}x wider")

# (2) coverage across the resonance
print("\nCoverage across the resonance (empirical vs nominal):")
cov_tab = {}
for lv in LEVELS:
    lo_p, hi_p = (1-lv)/2*100, (1+lv)/2*100
    cells = []; cov_tab[lv] = {}
    for dd in range(6):
        qlo = np.percentile(samp[:, :, dd], lo_p, 1); qhi = np.percentile(samp[:, :, dd], hi_p, 1)
        c = float(np.mean((theta[:, dd] >= qlo) & (theta[:, dd] <= qhi)))
        cov_tab[lv][NAMES[dd]] = round(c, 3); cells.append(f"{c*100:5.1f}%")
    print(f"  {lv*100:5.0f}% | " + " | ".join(cells))
calib_err = float(np.mean([abs(cov_tab[lv][n]-lv) for lv in LEVELS for n in NAMES]))
print(f"  mean |coverage-nominal| = {calib_err*100:.1f}%")

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax1.errorbar(grid, m2w, yerr=m2w_err, fmt="o-", color="#c0392b", lw=2, ms=6)
ax1.axvline(2.0, color="0.4", ls=":", lw=1.3)
ax1.set_xlabel("period ratio"); ax1.set_ylabel("m2 posterior / prior width")
ax1.set_title("Near-resonance mass degeneracy: posterior widens at the 2:1 separatrix")
ax1.grid(alpha=0.25)
ax2.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
for dd in range(6):
    ax2.plot(LEVELS, [cov_tab[lv][NAMES[dd]] for lv in LEVELS], "o-", label=NAMES[dd])
ax2.set_xlabel("nominal level"); ax2.set_ylabel("empirical coverage")
ax2.set_title(f"Calibration holds across the resonance ({calib_err*100:.1f}%)")
ax2.legend(fontsize=8, ncol=4); ax2.grid(alpha=0.25)
fig.suptitle("Cross-resonance hybrid (SBI trained on jaxttv, ratio 1.9-2.2)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("jaxttv_resonance.png", dpi=145)
print("wrote jaxttv_resonance.png")

json.dump({"grid": grid.tolist(), "m2_width": m2w.tolist(),
           "sep_width": round(float(m2w[sep]), 4), "offres_width": round(float(m2w[off]), 4),
           "collapse_factor": round(float(m2w[sep]/m2w[off]), 3),
           "coverage": cov_tab, "mean_calib_error": round(calib_err, 4)},
          open("_jaxttv_resonance.json", "w"), indent=2)
print("wrote _jaxttv_resonance.json")
