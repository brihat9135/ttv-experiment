"""
Real Kepler-9 fit, stage 2 (torch env): train the flow head on the config-specific jaxttv data
(kepler9_train.npz) and apply it to the REAL Kepler-9 O-C, inferring the planet masses/eccentricities
and comparing to the published dynamical masses (43.4 / 29.8 Me, Borsato 2019). Timing-only.

Also checks the config-specific flow is calibrated on held-out sims. Writes kepler9_fit.png + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flow import NPEFlow

SEED = 0
NOISE_MIN = 1.0                      # real Kepler-9 median timing error ~1 min
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]
NAMES = ["m1 (Kep-9b)", "m2 (Kep-9c)", "h1", "k1", "h2", "k2"]

d = np.load("kepler9_train.npz")
X_tr, th_tr = d["feats"], d["theta"]
X_va, th_va = d["feats_val"], d["theta_val"]
real = d["real_feat"]; published = d["published"]; IN = X_tr.shape[1]
print(f"in_dim {IN} | train {len(th_tr)} | published masses {published} Me")

f_mean, f_std = X_tr.mean(0), X_tr.std(0) + 1e-8
th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
rng = np.random.default_rng(SEED + 99)


def prep(X, th):
    x = (X + rng.normal(0, NOISE_MIN, X.shape) - f_mean) / f_std
    return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
            torch.tensor((th - th_mean) / th_std, dtype=torch.float32).to(DEVICE))


Xv, Yv = prep(X_va, th_va)
torch.manual_seed(SEED)
flow = NPEFlow(in_dim=IN, theta_dim=6).to(DEVICE)
opt = torch.optim.Adam(flow.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5)
best, best_state = np.inf, None
for ep in range(EPOCHS):
    flow.train(); Xt, Yt = prep(X_tr, th_tr); p = torch.randperm(len(Xt)); Xt, Yt = Xt[p], Yt[p]
    for i in range(0, len(Xt), BATCH):
        opt.zero_grad(); loss = flow.nll(Xt[i:i+BATCH], Yt[i:i+BATCH]); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
    sch.step(); flow.eval()
    with torch.no_grad():
        v = flow.nll(Xv, Yv).item()
    if v < best:
        best, best_state = v, {k: t.clone() for k, t in flow.state_dict().items()}
    if ep % 50 == 0 or ep == EPOCHS-1:
        print(f"  epoch {ep:3d} val NLL {v:+.3f} (best {best:+.3f})", flush=True)
flow.load_state_dict(best_state)
torch.save(flow.state_dict(), "kepler9_flow.pt")

# calibration on held-out sims
with torch.no_grad():
    sv = flow.sample(Xv, n=500).cpu().numpy() * th_std + th_mean
cov = {}
for lv in LEVELS:
    lo, hi = (1-lv)/2*100, (1+lv)/2*100
    cov[lv] = float(np.mean([np.mean((th_va[:, dd] >= np.percentile(sv[:, :, dd], lo, 1)) &
                                     (th_va[:, dd] <= np.percentile(sv[:, :, dd], hi, 1))) for dd in range(6)]))
calib = float(np.mean([abs(cov[lv]-lv) for lv in LEVELS]))

# ---- the real fit ----
xr = torch.tensor((real - f_mean) / f_std, dtype=torch.float32, device=DEVICE).view(1, -1)
with torch.no_grad():
    post = flow.sample(xr, n=20000).cpu().numpy()[0] * th_std + th_mean
m1, m2 = post[:, 0], post[:, 1]
print(f"\n  Kepler-9 REAL fit (timing-only):")
print(f"    m1 (9b): {m1.mean():.1f} +/- {m1.std():.1f} Me   (published 43.4)")
print(f"    m2 (9c): {m2.mean():.1f} +/- {m2.std():.1f} Me   (published 29.8)")
print(f"    e1: {np.hypot(post[:,2],post[:,3]).mean():.3f}  e2: {np.hypot(post[:,4],post[:,5]).mean():.3f}")
print(f"    config-specific flow calibration: {calib*100:.1f}%")
z1 = abs(m1.mean()-published[0])/m1.std(); z2 = abs(m2.mean()-published[1])/m2.std()
print(f"    offset from published: {z1:.1f} sigma (m1), {z2:.1f} sigma (m2)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax1.scatter(m1, m2, s=4, alpha=0.15, color="#1f6fb2", rasterized=True)
ax1.scatter([published[0]], [published[1]], marker="*", s=320, color="gold",
            edgecolor="k", zorder=5, label="published (Borsato 2019)")
ax1.axvline(published[0], color="0.5", ls=":", lw=1); ax1.axhline(published[1], color="0.5", ls=":", lw=1)
ax1.set_xlabel("m1  Kepler-9b  [Me]"); ax1.set_ylabel("m2  Kepler-9c  [Me]")
ax1.set_title("Kepler-9 masses from REAL TTVs (flow posterior)"); ax1.legend(fontsize=9)
for dd, (val, pub) in enumerate([(m1, published[0]), (m2, published[1])]):
    ax2.hist(val, bins=50, density=True, alpha=0.5, label=NAMES[dd])
    ax2.axvline(pub, color="k", ls=":", lw=1.2)
ax2.set_xlabel("mass [Me]"); ax2.set_yticks([]); ax2.legend(fontsize=9)
ax2.set_title(f"Marginals vs published (dotted); flow calib {calib*100:.1f}%")
fig.suptitle("Kepler-9: amortized flow posterior on REAL Kepler transit timings (timing-only)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("kepler9_fit.png", dpi=145)
print("  wrote kepler9_fit.png")

json.dump({"m1_mean": round(float(m1.mean()), 2), "m1_std": round(float(m1.std()), 2),
           "m2_mean": round(float(m2.mean()), 2), "m2_std": round(float(m2.std()), 2),
           "published": published.tolist(), "z_m1": round(float(z1), 2), "z_m2": round(float(z2), 2),
           "e1_mean": round(float(np.hypot(post[:,2],post[:,3]).mean()), 3),
           "e2_mean": round(float(np.hypot(post[:,4],post[:,5]).mean()), 3),
           "flow_calib": round(calib, 4), "in_dim": int(IN)},
          open("_kepler9_fit.json", "w"), indent=2)
print("  wrote _kepler9_fit.json")
