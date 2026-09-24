"""
Multi-seed the in-hybrid RV result, stage 2 (torch env): for each of 3 independent training sets,
train both arms (timing, timing+RV), evaluate the separatrix m2 tightening on the SHARED held-out
test set, and report mean +/- std. Puts an error bar on the ~83% headline. Writes
jaxttv_resrv_multiseed.png + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import MDN

SEEDS = [0, 1, 2]
NOISE_MIN, RV_NOISE = 0.5, 1.0
NOC, NRV = 60, 30
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]
NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]
PRIOR_STD = np.array([(15-3)/np.sqrt(12), (45-8)/np.sqrt(12), .075, .075, .075, .075])

te = np.load("jaxttv_resrv_test.npz")
t_oc, t_rv, t_ratio, t_theta = te["oc"], te["rv"], te["ratio"], te["theta"]
tn = np.random.default_rng(7)
t_oc_n = t_oc + tn.normal(0, NOISE_MIN, t_oc.shape)
t_rv_n = t_rv + tn.normal(0, RV_NOISE, t_rv.shape)
grid = np.unique(t_ratio)
sep = int(np.argmin(np.abs(grid - 2.0)))


def build_X(oc, rv, ratio, use_rv):
    parts = [oc] + ([rv] if use_rv else []) + [ratio[:, None]]
    return np.concatenate(parts, axis=1)


def noise_vec(ncols, use_rv):
    v = np.zeros(ncols); v[:NOC] = NOISE_MIN
    if use_rv:
        v[NOC:NOC+NRV] = RV_NOISE
    return v


def train_eval(seed, use_rv):
    tr = np.load(f"jaxttv_resrv_train_s{seed}.npz"); va = np.load(f"jaxttv_resrv_val_s{seed}.npz")
    Xtr = build_X(tr["oc"], tr["rv"], tr["ratio"], use_rv); th_tr = tr["theta"]
    Xva = build_X(va["oc"], va["rv"], va["ratio"], use_rv); th_va = va["theta"]
    IN = Xtr.shape[1]
    torch.manual_seed(seed)
    f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    nv = noise_vec(IN, use_rv); rng = np.random.default_rng(seed + 99)

    def prep(X, th):
        x = (X + rng.normal(0, 1, X.shape) * nv - f_mean) / f_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor((th - th_mean) / th_std, dtype=torch.float32).to(DEVICE))

    Xv, Yv = prep(Xva, th_va)
    net = MDN(in_dim=IN, theta_dim=6).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        Xt, Yt = prep(Xtr, th_tr); perm = torch.randperm(len(Xt)); Xt, Yt = Xt[perm], Yt[perm]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad(); loss = net.nll(Xt[i:i+BATCH], Yt[i:i+BATCH]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            v = net.nll(Xv, Yv).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)

    # evaluate on shared test
    Xte = build_X(t_oc_n, t_rv_n, t_ratio, use_rv)
    x = torch.tensor((Xte - f_mean) / f_std, dtype=torch.float32, device=DEVICE)
    samp = np.empty((len(Xte), 500, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            samp[i:i+1000] = net.sample(x[i:i+1000], n=500).cpu().numpy() * th_std + th_mean
    wcurve = np.array([samp[t_ratio == r][:, :, 1].std(1).mean() / PRIOR_STD[1] for r in grid])
    cov = {}
    for lv in LEVELS:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        cov[lv] = float(np.mean([np.mean((t_theta[:, dd] >= np.percentile(samp[:, :, dd], lo, 1)) &
                                          (t_theta[:, dd] <= np.percentile(samp[:, :, dd], hi, 1)))
                                 for dd in range(6)]))
    calib = float(np.mean([abs(cov[lv] - lv) for lv in LEVELS]))
    return wcurve, best, calib


Wt, Wr, tighten, nll_t, nll_r, cal_t, cal_r = [], [], [], [], [], [], []
for s in SEEDS:
    wt, bt, ct = train_eval(s, False)
    wr, br, cr = train_eval(s, True)
    Wt.append(wt); Wr.append(wr); nll_t.append(bt); nll_r.append(br); cal_t.append(ct); cal_r.append(cr)
    tg = (1 - wr[sep] / wt[sep]) * 100; tighten.append(tg)
    print(f"seed {s}: separatrix tightening {tg:.0f}%  | valNLL {bt:+.2f}->{br:+.2f} | "
          f"calib {ct*100:.1f}%->{cr*100:.1f}%", flush=True)

Wt, Wr = np.array(Wt), np.array(Wr)
tighten = np.array(tighten)
print(f"\n=== {len(SEEDS)} seeds ===")
print(f"  separatrix m2 tightening: {tighten.mean():.0f} +/- {tighten.std():.0f} %")
print(f"  val NLL timing {np.mean(nll_t):+.2f}+/-{np.std(nll_t):.2f}  ->  "
      f"timing+RV {np.mean(nll_r):+.2f}+/-{np.std(nll_r):.2f}")
print(f"  calibration timing {np.mean(cal_t)*100:.1f}%  timing+RV {np.mean(cal_r)*100:.1f}%")

fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.plot(grid, Wt.mean(0), "-", color="#c0392b", lw=2, label="timing only")
ax.fill_between(grid, Wt.mean(0)-Wt.std(0), Wt.mean(0)+Wt.std(0), color="#c0392b", alpha=0.2)
ax.plot(grid, Wr.mean(0), "-", color="#1f6fb2", lw=2, label="timing + RV")
ax.fill_between(grid, Wr.mean(0)-Wr.std(0), Wr.mean(0)+Wr.std(0), color="#1f6fb2", alpha=0.2)
ax.axvline(2.0, color="0.4", ls=":", lw=1.3)
ax.annotate(f"{tighten.mean():.0f} $\\pm$ {tighten.std():.0f}% tighter\nat separatrix",
            (2.0, (Wt.mean(0)[sep]+Wr.mean(0)[sep])/2), fontsize=10, ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.set_xlabel("period ratio"); ax.set_ylabel("m2 posterior / prior width")
ax.set_title(f"In-hybrid RV degeneracy-break, {len(SEEDS)} seeds (band = std across seeds)")
ax.legend(); ax.grid(alpha=0.25); ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig("jaxttv_resrv_multiseed.png", dpi=145)
print("wrote jaxttv_resrv_multiseed.png")

json.dump({"seeds": SEEDS, "tighten_pct_per_seed": tighten.tolist(),
           "tighten_mean": round(float(tighten.mean()), 1), "tighten_std": round(float(tighten.std()), 1),
           "valnll_timing": [round(x, 3) for x in nll_t], "valnll_timing_rv": [round(x, 3) for x in nll_r],
           "calib_timing": round(float(np.mean(cal_t)), 4), "calib_timing_rv": round(float(np.mean(cal_r)), 4)},
          open("_jaxttv_resrv_multiseed.json", "w"), indent=2)
print("wrote _jaxttv_resrv_multiseed.json")
