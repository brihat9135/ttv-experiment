"""
Kepler-9 durations, STEP A (torch env, REBOUND): a self-consistent SIMULATION demonstration of
what transit durations buy at Kepler-9's geometry. jaxttv has no durations, so we use our REBOUND
simulator (simulator.py) reconfigured to Kepler-9's real PERIODS (19.246/38.950 d), and compare
two amortized flows on the same systems:
    timing-only        (97-D O-C)
    timing + durations (97 O-C + 97 duration anomalies = 194-D)
The timing-only fit reproduces the mass-ratio RIDGE we saw on the REAL data (kepler9_fit.py);
adding durations should LOCALIZE the mass along it. This is the honest "what durations would do
for Kepler-9" step, self-consistent so no duration-realism confound. (Step B = real TDV fit.)

Note: uses a representative phase (PHASE2 default), not Kepler-9's exact resonant phase, so absolute
TTV amplitudes are illustrative; the durations-vs-timing comparison is the point. Writes
kepler9_durations_sim.png + json.
"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulator as S
from flow import NPEFlow

# ---- reconfigure REBOUND for Kepler-9 geometry ----
S.P1 = 19.2463 * S.DAY
S.BASELINE = 1560.0 * S.DAY
S.STEP = S.P1 / 60.0
S.N_TRANSITS_1, S.N_TRANSITS_2 = 64, 33
S.FEATURE_DIM = 97; S.DURATION_DIM = 97; S.FEATURE_DIM_FULL = 194
PC = 38.9498 * S.DAY
NT = S.N_TRANSITS_1 + S.N_TRANSITS_2            # 97
M_LO, M_HI, EMAX = 5.0, 80.0, 0.15
TIME_NOISE, DUR_NOISE = 1.0, 3.0
N_TRAIN, N_VAL = 16000, 3000
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIOR_STD = np.array([(M_HI-M_LO)/np.sqrt(12), (M_HI-M_LO)/np.sqrt(12), .075, .075, .075, .075])
PUBLISHED = np.array([43.4, 29.8])
THETA_TEST = np.array([43.4, 29.8, 0.06, 0.0, 0.05, 0.03])   # published masses, illustrative ecc


def sample_prior(n, rng):
    m1 = rng.uniform(M_LO, M_HI, n); m2 = rng.uniform(M_LO, M_HI, n)
    e1 = EMAX*np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2*np.pi, n)
    e2 = EMAX*np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2*np.pi, n)
    return np.stack([m1, m2, e1*np.cos(w1), e1*np.sin(w1), e2*np.cos(w2), e2*np.sin(w2)], 1)


def gen(n, seed):
    rng = np.random.default_rng(seed); th = sample_prior(n, rng)
    f = np.full((n, 194), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i+2000, n))
        f[sl] = S.simulate(th[sl], p2=np.full(sl.stop-sl.start, PC), durations=True)
    ok = ~np.isnan(f).any(1)
    return th[ok], f[ok]


def train(feat_cols, Xtr_raw, th_tr, Xva_raw, th_va, seed=0):
    Xtr, Xva = Xtr_raw[:, feat_cols], Xva_raw[:, feat_cols]
    nv = np.where(np.array(feat_cols) < NT, TIME_NOISE, DUR_NOISE)
    f_mean, f_std = Xtr.mean(0), Xtr.std(0)+1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0)+1e-8
    rng = np.random.default_rng(seed+99)

    def prep(X, th):
        x = (X + rng.normal(0, 1, X.shape)*nv - f_mean)/f_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor((th-th_mean)/th_std, dtype=torch.float32).to(DEVICE))

    Xv, Yv = prep(Xva, th_va)
    torch.manual_seed(seed)
    flow = NPEFlow(in_dim=len(feat_cols), theta_dim=6).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=LR); sch = torch.optim.lr_scheduler.StepLR(opt, 100, .5)
    best, bs = np.inf, None
    for ep in range(EPOCHS):
        flow.train(); Xt, Yt = prep(Xtr, th_tr); p = torch.randperm(len(Xt)); Xt, Yt = Xt[p], Yt[p]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad(); l = flow.nll(Xt[i:i+BATCH], Yt[i:i+BATCH]); l.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        sch.step(); flow.eval()
        with torch.no_grad():
            v = flow.nll(Xv, Yv).item()
        if v < best:
            best, bs = v, {k: t.clone() for k, t in flow.state_dict().items()}
    flow.load_state_dict(bs)
    return flow, (f_mean, f_std, th_mean, th_std, nv), best


def posterior(flow, stats, feat_cols, feat_raw):
    f_mean, f_std, th_mean, th_std, nv = stats
    x = torch.tensor((feat_raw[feat_cols] - f_mean)/f_std, dtype=torch.float32, device=DEVICE).view(1, -1)
    with torch.no_grad():
        return flow.sample(x, n=15000).cpu().numpy()[0]*th_std + th_mean


print(f"Device: {DEVICE}. Kepler-9 durations Step A (REBOUND @ Kepler-9 periods).")
th_tr, F_tr = gen(N_TRAIN, 0); th_va, F_va = gen(N_VAL, 1)
print(f"  train {len(th_tr)}, val {len(th_va)} (usable after stability cut)")

cols_t = list(range(NT))              # timing only
cols_td = list(range(194))            # timing + durations
flow_t, st_t, nll_t = train(cols_t, F_tr, th_tr, F_va, th_va)
flow_td, st_td, nll_td = train(cols_td, F_tr, th_tr, F_va, th_va)
print(f"  val NLL: timing {nll_t:+.2f} -> timing+dur {nll_td:+.2f}")

# test system at published masses
rng = np.random.default_rng(7)
clean = S.simulate(THETA_TEST[None, :], p2=np.array([PC]), durations=True)[0]
obs = clean.copy()
obs[:NT] += rng.normal(0, TIME_NOISE, NT); obs[NT:] += rng.normal(0, DUR_NOISE, 194-NT)
pt = posterior(flow_t, st_t, cols_t, obs)
ptd = posterior(flow_td, st_td, cols_td, obs)
w_t = pt[:, :2].std(0)/PRIOR_STD[:2]; w_td = ptd[:, :2].std(0)/PRIOR_STD[:2]
tighten = (1 - w_td/w_t)*100
print(f"  test @ published masses: m1 tighten {tighten[0]:.0f}%, m2 tighten {tighten[1]:.0f}%")
print(f"    m1: timing {pt[:,0].mean():.1f}+/-{pt[:,0].std():.1f} -> +dur {ptd[:,0].mean():.1f}+/-{ptd[:,0].std():.1f} (pub 43.4)")
print(f"    m2: timing {pt[:,1].mean():.1f}+/-{pt[:,1].std():.1f} -> +dur {ptd[:,1].mean():.1f}+/-{ptd[:,1].std():.1f} (pub 29.8)")

fig, ax = plt.subplots(figsize=(7.4, 6.4))
ax.scatter(pt[:, 0], pt[:, 1], s=4, alpha=0.12, color="#c0392b", label="timing only (ridge)", rasterized=True)
ax.scatter(ptd[:, 0], ptd[:, 1], s=4, alpha=0.2, color="#1f6fb2", label="timing + durations (localized)", rasterized=True)
ax.scatter([PUBLISHED[0]], [PUBLISHED[1]], marker="*", s=340, color="gold", edgecolor="k", zorder=5, label="published")
ax.set_xlabel("m1  Kepler-9b  [Me]"); ax.set_ylabel("m2  Kepler-9c  [Me]")
ax.set_title(f"Step A (simulation @ Kepler-9 periods): durations localize the mass ridge\n"
             f"m1 {tighten[0]:.0f}% / m2 {tighten[1]:.0f}% tighter", fontsize=11)
ax.legend(fontsize=9, markerscale=2); ax.set_xlim(0, 90); ax.set_ylim(0, 90)
fig.tight_layout(); fig.savefig("kepler9_durations_sim.png", dpi=145)
print("  wrote kepler9_durations_sim.png")

import json
json.dump({"m1_tighten_pct": round(float(tighten[0]), 1), "m2_tighten_pct": round(float(tighten[1]), 1),
           "nll_timing": round(float(nll_t), 3), "nll_timing_dur": round(float(nll_td), 3),
           "test_theta": THETA_TEST.tolist(),
           "m1_timing": [round(float(pt[:,0].mean()),1), round(float(pt[:,0].std()),1)],
           "m1_dur": [round(float(ptd[:,0].mean()),1), round(float(ptd[:,0].std()),1)],
           "m2_timing": [round(float(pt[:,1].mean()),1), round(float(pt[:,1].std()),1)],
           "m2_dur": [round(float(ptd[:,1].mean()),1), round(float(ptd[:,1].std()),1)]},
          open("_kepler9_durations_sim.json", "w"), indent=2)
print("  wrote _kepler9_durations_sim.json")
