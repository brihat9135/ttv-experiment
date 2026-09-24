"""
Break the near-resonance degeneracy in-hybrid, stage 3 (torch env): compare the two arms
(timing vs timing+RV) on the held-out cross-resonance test set. The headline: at the 2:1
separatrix the m2 mass posterior is WIDE with timing alone and TIGHT once RV is added, the
jaxttv analog of the REBOUND RV result (16%->83%). Also reports per-arm calibration.
Writes jaxttv_resonance_rv.png + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import MDN

NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]
PRIOR_STD = np.array([(15-3)/np.sqrt(12), (45-8)/np.sqrt(12), 0.075, 0.075, 0.075, 0.075])
NOISE_MIN, RV_NOISE = 0.5, 1.0
NOC, NRV = 60, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]

te = np.load("jaxttv_resrv_test.npz")
theta, ratio, oc, rv = te["theta"], te["ratio"], te["oc"], te["rv"]
rng = np.random.default_rng(7)
oc_n = oc + rng.normal(0, NOISE_MIN, oc.shape)
rv_n = rv + rng.normal(0, RV_NOISE, rv.shape)


def infer(arm, X):
    norm = np.load(f"norm_resrv_{arm}.npz")
    net = MDN(in_dim=X.shape[1], theta_dim=6).to(DEVICE)
    net.load_state_dict(torch.load(f"mdn_resrv_{arm}.pt", map_location=DEVICE)); net.eval()
    x = torch.tensor((X - norm["f_mean"]) / norm["f_std"], dtype=torch.float32, device=DEVICE)
    out = np.empty((len(X), 1000, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            out[i:i+1000] = net.sample(x[i:i+1000], n=1000).cpu().numpy() * norm["th_std"] + norm["th_mean"]
    return out


X_t = np.concatenate([oc_n, ratio[:, None]], axis=1)
X_r = np.concatenate([oc_n, rv_n, ratio[:, None]], axis=1)
s_t, s_r = infer("timing", X_t), infer("timing_rv", X_r)

grid = np.unique(ratio)


def width_curve(samp):
    w, e = [], []
    for r in grid:
        m = ratio == r
        ww = samp[m][:, :, 1].std(1) / PRIOR_STD[1]
        w.append(ww.mean()); e.append(ww.std() / np.sqrt(m.sum()))
    return np.array(w), np.array(e)


wt, et = width_curve(s_t); wr, er = width_curve(s_r)
sep = np.argmin(np.abs(grid - 2.0))
tighten = (1 - wr[sep] / wt[sep]) * 100
print(f"At the separatrix (r={grid[sep]:.3f}): m2 width  timing {wt[sep]:.3f} -> timing+RV {wr[sep]:.3f}")
print(f"  => near-resonance m2 tightening from RV: {tighten:.0f}%")


def coverage(samp):
    tab = {}
    for lv in LEVELS:
        lo_p, hi_p = (1-lv)/2*100, (1+lv)/2*100; tab[lv] = {}
        for dd in range(6):
            qlo = np.percentile(samp[:, :, dd], lo_p, 1); qhi = np.percentile(samp[:, :, dd], hi_p, 1)
            tab[lv][NAMES[dd]] = round(float(np.mean((theta[:, dd] >= qlo) & (theta[:, dd] <= qhi))), 3)
    err = float(np.mean([abs(tab[lv][n]-lv) for lv in LEVELS for n in NAMES]))
    return tab, err


cov_t, err_t = coverage(s_t); cov_r, err_r = coverage(s_r)
print(f"  calibration: timing {err_t*100:.1f}%  |  timing+RV {err_r*100:.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax1.errorbar(grid, wt, yerr=et, fmt="o-", color="#c0392b", lw=2, ms=5, label="timing only")
ax1.errorbar(grid, wr, yerr=er, fmt="s-", color="#1f6fb2", lw=2, ms=5, label="timing + RV")
ax1.axvline(2.0, color="0.4", ls=":", lw=1.3)
ax1.annotate(f"{tighten:.0f}% tighter\nat separatrix", (2.0, (wt[sep]+wr[sep])/2),
             fontsize=9, ha="center", color="#333",
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax1.set_xlabel("period ratio"); ax1.set_ylabel("m2 posterior / prior width")
ax1.set_title("RV breaks the near-resonance mass degeneracy (in-hybrid)")
ax1.legend(); ax1.grid(alpha=0.25); ax1.set_ylim(0, None)
ax2.plot([0, 1], [0, 1], "k--", lw=1)
for dd in range(6):
    ax2.plot(LEVELS, [cov_r[lv][NAMES[dd]] for lv in LEVELS], "o-", label=NAMES[dd])
ax2.set_xlabel("nominal level"); ax2.set_ylabel("empirical coverage")
ax2.set_title(f"timing+RV arm stays calibrated ({err_r*100:.1f}%)")
ax2.legend(fontsize=8, ncol=4); ax2.grid(alpha=0.25)
fig.suptitle("Cross-resonance hybrid: RV pins the mass at the 2:1 separatrix", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("jaxttv_resonance_rv.png", dpi=145)
print("wrote jaxttv_resonance_rv.png")

json.dump({"grid": grid.tolist(), "m2_width_timing": wt.tolist(), "m2_width_timing_rv": wr.tolist(),
           "sep_ratio": float(grid[sep]), "sep_width_timing": round(float(wt[sep]), 4),
           "sep_width_timing_rv": round(float(wr[sep]), 4), "sep_tightening_pct": round(float(tighten), 1),
           "calib_timing": round(err_t, 4), "calib_timing_rv": round(err_r, 4)},
          open("_jaxttv_resonance_rv.json", "w"), indent=2)
print("wrote _jaxttv_resonance_rv.json")
