"""
Kepler-9 durations, STEP B (torch env, REBOUND): fit the REAL Holczer data with timing + real
transit-duration variations (TDV). Uses REBOUND at Kepler-9's real periods with the resonant PHASE
found by the gating diagnostic (~1.05 rad, where REBOUND reproduces the real O-C pattern, corr~1).
Two flows on the same real system: timing-only vs timing+TDV, to see the mass localize on REAL data.

Honest caveats (documented): (1) fixed phase (not jointly inferred); (2) our duration model is
edge-on, 1 Rsun (a realism gap vs the true geometry) — so the TDV arm is more approximate than
timing; (3) single-seed. Feature: per-transit O-C (min) + fractional TDV, at the transits where BOTH
are clean. Sign of the real O-C/TDV is aligned to REBOUND's convention via the published-mass model.
Writes kepler9_stepB.png + json.
"""
import csv, json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulator as S
from flow import NPEFlow

# ---- configure REBOUND at Kepler-9 ----
S.P1 = 19.2463 * S.DAY; S.BASELINE = 1560.0 * S.DAY; S.STEP = S.P1 / 60.0
S.PHASE2 = 1.05                                  # resonant phase from the gating scan
PC = 38.9498 * S.DAY
PUBLISHED = np.array([43.4, 29.8])
M_LO, M_HI, EMAX = 5.0, 80.0, 0.15
TIME_NOISE = 1.0
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRIOR_STD = np.array([(M_HI-M_LO)/np.sqrt(12)]*2 + [.075]*4)


def load_col(path, col):                          # col index in the TSV (2=OC or 2=TDV)
    if not __import__("os").path.exists(path):    # reproducible: fetch real TDV from VizieR if absent
        koi = path.split("_")[-1].replace(".tsv", "")
        url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/225/9/table3"
               f"&KOI={koi}&-out=KOI,N,TDV,e_TDV,f_TDV,Out,Over&-out.max=99999")
        __import__("urllib.request", fromlist=["request"]).urlretrieve(url, path)
    d = {}
    for line in open(path):
        if (line[:1].isdigit() or line[:1] == " ") and "\t" in line:
            p = line.rstrip().split("\t")
            if len(p) >= 7 and p[1].strip() and p[2].strip():
                try:
                    n = int(p[1]); v = float(p[2]); e = float(p[3])
                    flag = p[4].strip(); out = p[5].strip(); over = p[6].strip()
                    if flag == "" and out in ("", "0") and over in ("", "0"):
                        d[n] = (v, e)
                except ValueError:
                    pass
    return d


# real O-C (from csv) and TDV (from tsv), intersect clean transits per planet
ocsv = {}
for r in csv.DictReader(open("kepler9_ttv.csv")):
    ocsv.setdefault(r["planet"], {})[int(r["N"])] = float(r["OC_min"])
tdv = {"Kepler-9b": load_col("kepler9_tdv_377.01.tsv", 2), "Kepler-9c": load_col("kepler9_tdv_377.02.tsv", 2)}
seg = {}
for p, koi in [("Kepler-9b", 0), ("Kepler-9c", 1)]:
    Ns = sorted(set(ocsv[p]) & set(tdv[p]))
    seg[p] = dict(N=np.array(Ns), oc=np.array([ocsv[p][n] for n in Ns]),
                  tdv=np.array([tdv[p][n][0] for n in Ns]))
Nb, Nc = seg["Kepler-9b"]["N"], seg["Kepler-9c"]["N"]
nb, nc = len(Nb), len(Nc)
S.N_TRANSITS_1, S.N_TRANSITS_2 = int(Nb.max())+1, int(Nc.max())+1
S.FEATURE_DIM = S.N_TRANSITS_1 + S.N_TRANSITS_2
S.DURATION_DIM = S.FEATURE_DIM; S.FEATURE_DIM_FULL = 2 * S.FEATURE_DIM
CIRC = np.array([S._circ_duration(S.P1), S._circ_duration(PC)])   # years, per planet
print(f"clean-both transits: 9b {nb}, 9c {nc}")


def sim_feature(theta):                           # -> (B, nb+nc) O-C(min) , (B, nb+nc) fractional TDV
    f = S.simulate(theta, p2=np.full(len(theta), PC), durations=True)
    n1 = S.N_TRANSITS_1
    oc_full, dur_full = f[:, :S.FEATURE_DIM], f[:, S.FEATURE_DIM:]     # dur = anomaly in minutes
    ocb = oc_full[:, Nb]; occ = oc_full[:, Nc]
    # fractional TDV from the minute-anomaly feature: frac = (anom-mean)*MIN / circ_dur
    db = dur_full[:, Nb]; dc = dur_full[:, Nc]
    fb = (db - np.nanmean(db, 1, keepdims=True)) * S.MIN / CIRC[0]
    fc = (dc - np.nanmean(dc, 1, keepdims=True)) * S.MIN / CIRC[1]
    return np.concatenate([ocb, occ], 1), np.concatenate([fb, fc], 1)


def sample_prior(n, rng):
    m1 = rng.uniform(M_LO, M_HI, n); m2 = rng.uniform(M_LO, M_HI, n)
    e1 = EMAX*np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2*np.pi, n)
    e2 = EMAX*np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2*np.pi, n)
    return np.stack([m1, m2, e1*np.cos(w1), e1*np.sin(w1), e2*np.cos(w2), e2*np.sin(w2)], 1)


def gen(n, seed):
    rng = np.random.default_rng(seed); th = sample_prior(n, rng)
    OC = np.full((n, nb+nc), np.nan); TD = np.full((n, nb+nc), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i+2000, n))
        oc, td = sim_feature(th[sl]); OC[sl] = oc; TD[sl] = td
    ok = ~np.isnan(OC).any(1) & ~np.isnan(TD).any(1)
    return th[ok], OC[ok], TD[ok]


# ---- sign alignment + realism check via the published-mass model ----
pub_oc, pub_td = sim_feature(np.array([[PUBLISHED[0], PUBLISHED[1], 0.10, 0.0, 0.07, 0.03]]))
real_oc = np.concatenate([seg["Kepler-9b"]["oc"], seg["Kepler-9c"]["oc"]])
real_td = np.concatenate([seg["Kepler-9b"]["tdv"], seg["Kepler-9c"]["tdv"]])
real_td = real_td - real_td.mean()
s_oc = np.sign(np.corrcoef(pub_oc[0], real_oc)[0, 1])
s_td = np.sign(np.corrcoef(pub_td[0], real_td)[0, 1])
real_oc = real_oc * s_oc; real_td = real_td * s_td
print(f"sign align: O-C x{s_oc:+.0f} (corr {abs(np.corrcoef(pub_oc[0],real_oc)[0,1]):.2f}), "
      f"TDV x{s_td:+.0f} (corr {abs(np.corrcoef(pub_td[0],real_td)[0,1]):.2f})")
print(f"TDV realism: model frac-TDV rms {pub_td.std():.4f} vs real {real_td.std():.4f}")

TDV_NOISE = float(np.median([tdv["Kepler-9b"][n][1] for n in Nb] + [tdv["Kepler-9c"][n][1] for n in Nc]))
th_tr, OC_tr, TD_tr = gen(16000, 0); th_va, OC_va, TD_va = gen(3000, 1)
print(f"train {len(th_tr)}, val {len(th_va)} | TDV noise {TDV_NOISE:.4f}")


def train_fit(use_tdv):
    Xtr = np.concatenate([OC_tr, TD_tr], 1) if use_tdv else OC_tr
    Xva = np.concatenate([OC_va, TD_va], 1) if use_tdv else OC_va
    real = np.concatenate([real_oc, real_td]) if use_tdv else real_oc
    nv = np.concatenate([np.full(nb+nc, TIME_NOISE), np.full(nb+nc, TDV_NOISE)]) if use_tdv else np.full(nb+nc, TIME_NOISE)
    f_mean, f_std = Xtr.mean(0), Xtr.std(0)+1e-8; th_mean, th_std = th_tr.mean(0), th_tr.std(0)+1e-8
    rng = np.random.default_rng(99)

    def prep(X, th):
        x = (X + rng.normal(0, 1, X.shape)*nv - f_mean)/f_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor((th-th_mean)/th_std, dtype=torch.float32).to(DEVICE))

    Xv, Yv = prep(Xva, th_va); torch.manual_seed(0)
    flow = NPEFlow(in_dim=Xtr.shape[1], theta_dim=6).to(DEVICE)
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
        if v < best: best, bs = v, {k: t.clone() for k, t in flow.state_dict().items()}
    flow.load_state_dict(bs)
    xr = torch.tensor((real - f_mean)/f_std, dtype=torch.float32, device=DEVICE).view(1, -1)
    with torch.no_grad():
        post = flow.sample(xr, n=15000).cpu().numpy()[0]*th_std + th_mean
    return post, best


pt, nll_t = train_fit(False); ptd, nll_td = train_fit(True)
print(f"\nREAL Kepler-9 fit (REBOUND, phase {S.PHASE2}):  val NLL timing {nll_t:+.2f} -> +TDV {nll_td:+.2f}")
for i, nm, pub in [(0, "m1(9b)", 43.4), (1, "m2(9c)", 29.8)]:
    print(f"  {nm}: timing {pt[:,i].mean():.1f}+/-{pt[:,i].std():.1f} -> +TDV {ptd[:,i].mean():.1f}+/-{ptd[:,i].std():.1f} (pub {pub})")
tighten = (1 - ptd[:, :2].std(0)/pt[:, :2].std(0))*100
z = np.abs(ptd[:, :2].mean(0)-PUBLISHED)/ptd[:, :2].std(0)
print(f"  m1/m2 tightening {tighten[0]:.0f}%/{tighten[1]:.0f}% ; offset {z[0]:.1f}/{z[1]:.1f} sigma from published")

fig, ax = plt.subplots(figsize=(7.4, 6.4))
ax.scatter(pt[:, 0], pt[:, 1], s=4, alpha=0.12, color="#c0392b", label="timing only", rasterized=True)
ax.scatter(ptd[:, 0], ptd[:, 1], s=4, alpha=0.2, color="#1f6fb2", label="timing + real TDV", rasterized=True)
ax.scatter([PUBLISHED[0]], [PUBLISHED[1]], marker="*", s=340, color="gold", edgecolor="k", zorder=5, label="published")
ax.set_xlabel("m1  Kepler-9b  [Me]"); ax.set_ylabel("m2  Kepler-9c  [Me]"); ax.set_xlim(0, 90); ax.set_ylim(0, 90)
ax.set_title(f"Step B: REAL Kepler-9 timing + real TDV (m1 {tighten[0]:.0f}% / m2 {tighten[1]:.0f}% tighter)", fontsize=11)
ax.legend(fontsize=9, markerscale=2)
fig.tight_layout(); fig.savefig("kepler9_stepB.png", dpi=145); print("  wrote kepler9_stepB.png")
json.dump({"phase": S.PHASE2, "sign_oc": float(s_oc), "sign_tdv": float(s_td),
           "tdv_model_rms": round(float(pub_td.std()), 4), "tdv_real_rms": round(float(real_td.std()), 4),
           "m1_timing": [round(float(pt[:,0].mean()),1), round(float(pt[:,0].std()),1)],
           "m1_tdv": [round(float(ptd[:,0].mean()),1), round(float(ptd[:,0].std()),1)],
           "m2_timing": [round(float(pt[:,1].mean()),1), round(float(pt[:,1].std()),1)],
           "m2_tdv": [round(float(ptd[:,1].mean()),1), round(float(ptd[:,1].std()),1)],
           "tighten_pct": [round(float(tighten[0]),1), round(float(tighten[1]),1)],
           "sigma_from_published": [round(float(z[0]),2), round(float(z[1]),2)],
           "published": PUBLISHED.tolist()}, open("_kepler9_stepB.json", "w"), indent=2)
print("  wrote _kepler9_stepB.json")
