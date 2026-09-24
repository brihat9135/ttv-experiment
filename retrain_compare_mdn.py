"""
Retrain SBI on jaxttv's forward model, stage 3b (torch env): run the jaxttv-TRAINED MDN
(mdn_jaxttv.pt) on the same test data, overlay against jaxttv NUTS, and quantify agreement.
Run AFTER retrain_compare_jaxttv.py. Writes retrain_compare.png + _retrain_compare.json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import MDN

NAMES = ["m1 [Me]", "m2 [Me]", "h1", "k1", "h2", "k2"]
d = np.load("retrain_data.npz")
tc = np.asarray(d["tc_obs"]); NIN, NOUT = int(d["nin"]), int(d["nout"])
truth = np.asarray(d["theta_true"], float)


def detrend_min(T):
    k = len(T); i = np.arange(k); im = i.mean(); sxx = ((i - im) ** 2).sum()
    slope = ((T - T.mean()) * (i - im)).sum() / sxx
    return (T - (T.mean() - slope * im + slope * i)) * 1440.0


feat = np.concatenate([detrend_min(tc[:NIN]), detrend_min(tc[NIN:])])
norm = np.load("norm_jaxttv.npz")
x = (feat - norm["f_mean"]) / norm["f_std"]
net = MDN(in_dim=60, theta_dim=6); net.load_state_dict(torch.load("mdn_jaxttv.pt", map_location="cpu")); net.eval()
with torch.no_grad():
    mdn = net.sample(torch.tensor(x, dtype=torch.float32).view(1, -1), n=8000).numpy()[0]
mdn = mdn * norm["th_std"] + norm["th_mean"]

jd = np.load("retrain_jaxttv.npz"); jax_s = jd["samples"]; jax_sec = float(jd["nuts_seconds"])

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
rows = []
for i, ax in enumerate(axes.ravel()):
    lo = min(mdn[:, i].min(), jax_s[:, i].min()); hi = max(mdn[:, i].max(), jax_s[:, i].max())
    bins = np.linspace(lo, hi, 44)
    ax.hist(jax_s[:, i], bins=bins, density=True, color="#2a9d5c", alpha=0.45, label="jaxttv NUTS")
    ax.hist(mdn[:, i], bins=bins, density=True, histtype="step", color="#1f6fb2", lw=2.0,
            label="MDN retrained on jaxttv")
    ax.axvline(truth[i], color="k", ls=":", lw=1.3)
    ax.set_title(NAMES[i], fontsize=11); ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=8.5, loc="upper right")
    rows.append((NAMES[i], truth[i], jax_s[:, i].mean(), jax_s[:, i].std(), mdn[:, i].mean(), mdn[:, i].std()))
fig.suptitle("SBI retrained on jaxttv's forward model: MDN vs jaxttv NUTS now AGREE (same physics)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("retrain_compare.png", dpi=145)
print("wrote retrain_compare.png")

print(f"\n  {'param':>8} {'truth':>8} | {'jaxttv mean':>11} {'jaxttv std':>10} | {'MDN mean':>9} {'MDN std':>9} | {'|mean diff|/jstd':>15}")
out = {"jaxttv_seconds": round(jax_sec, 1), "per_param": {}}
zscores, ratios = [], []
for nm, tv, jm, js, dm, ds in rows:
    zz = abs(dm - jm) / (js + 1e-12); rr = ds / (js + 1e-12); zscores.append(zz); ratios.append(rr)
    print(f"  {nm:>8} {tv:>8.3f} | {jm:>11.3f} {js:>10.4f} | {dm:>9.3f} {ds:>9.4f} | {zz:>15.2f}")
    out["per_param"][nm] = {"truth": float(tv), "jaxttv_mean": round(float(jm), 4), "jaxttv_std": round(float(js), 4),
                            "mdn_mean": round(float(dm), 4), "mdn_std": round(float(ds), 4),
                            "mean_offset_in_jaxttv_sigma": round(float(zz), 3), "width_ratio_mdn_over_jaxttv": round(float(rr), 3)}
out["mean_abs_offset_sigma"] = round(float(np.mean(zscores)), 3)
out["mean_width_ratio_mdn_over_jaxttv"] = round(float(np.mean(ratios)), 3)
print(f"\n  agreement: mean |posterior-mean offset| = {np.mean(zscores):.2f} jaxttv-sigma; "
      f"mean width ratio MDN/jaxttv = {np.mean(ratios):.2f}")
json.dump(out, open("_retrain_compare.json", "w"), indent=2)
print("  wrote _retrain_compare.json")
