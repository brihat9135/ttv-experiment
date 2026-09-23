"""
Matched-system head-to-head, stage 3 (torch env): overlay the two posteriors that saw the
IDENTICAL transit-time data, our amortized MDN vs jaxttv NUTS, and quantify agreement.
Writes headtohead.png + _headtohead.json.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("headtohead_data.npz", allow_pickle=True)
truth = np.asarray(d["theta_true"], float)
names = [str(x) for x in d["theta_names"]]
mdn = np.load("headtohead_mdn.npz")["samples"]
jd = np.load("headtohead_jaxttv.npz")
jax_s = jd["samples"]
jax_sec = float(jd["nuts_seconds"])

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
rows = []
for i, ax in enumerate(axes.ravel()):
    lo = min(mdn[:, i].min(), jax_s[:, i].min())
    hi = max(mdn[:, i].max(), jax_s[:, i].max())
    bins = np.linspace(lo, hi, 44)
    ax.hist(jax_s[:, i], bins=bins, density=True, color="#6a51a3", alpha=0.45,
            label="jaxttv NUTS")
    ax.hist(mdn[:, i], bins=bins, density=True, histtype="step", color="#1f6fb2", lw=2.0,
            label="MDN (amortized)")
    ax.axvline(truth[i], color="k", ls=":", lw=1.3)
    ax.set_title(names[i], fontsize=11); ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=8.5, loc="upper right")
    rows.append((names[i], truth[i], jax_s[:, i].mean(), jax_s[:, i].std(),
                 mdn[:, i].mean(), mdn[:, i].std()))
fig.suptitle("Matched-system head-to-head: amortized MDN vs jaxttv NUTS on the SAME transit times",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("headtohead.png", dpi=145)
print("wrote headtohead.png")

print(f"\n  {'param':>8} {'truth':>8} | {'jaxttv mean':>11} {'jaxttv std':>10} | "
      f"{'MDN mean':>9} {'MDN std':>9} | {'std MDN/jaxttv':>14}")
ratios, out = [], {"jaxttv_seconds": round(jax_sec, 1), "per_param": {}}
for nm, tv, jm, js, dm, ds in rows:
    r = ds / (js + 1e-12); ratios.append(r)
    print(f"  {nm:>8} {tv:>8.3f} | {jm:>11.3f} {js:>10.4f} | {dm:>9.3f} {ds:>9.4f} | {r:>14.2f}")
    out["per_param"][nm] = {"truth": float(tv), "jaxttv_mean": round(float(jm), 4),
                            "jaxttv_std": round(float(js), 4), "mdn_mean": round(float(dm), 4),
                            "mdn_std": round(float(ds), 4), "std_ratio_mdn_over_jaxttv": round(float(r), 3)}
out["mean_std_ratio_mdn_over_jaxttv"] = round(float(np.mean(ratios)), 3)
print(f"\n  mean posterior-width ratio MDN/jaxttv = {np.mean(ratios):.2f}  "
      f"(1.0 = same width; >1 MDN wider/conservative)")
print(f"  cost: jaxttv NUTS {jax_sec:.0f} s per system  vs  MDN ~0.007 s per system")
json.dump(out, open("_headtohead.json", "w"), indent=2)
print("  wrote _headtohead.json")
