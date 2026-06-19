"""
The k-constraining sequel to observables_cross_resonance.py.

That experiment showed durations only PARTIALLY rescue the near-resonance informativeness
collapse: spanning the 2:1 (ratio ~ U(1.9,2.2), e_max=0.15) the mass posterior tightened
only ~22% at the separatrix, because a transit duration is a single mid-transit speed that
pins the e-vector component h (=e cos w) but leaves k (=e sin w) free -- and with k loose
the near-resonant mass stays degenerate.

Radial velocity is the missing lever. The star's line-of-sight reflex curve has amplitude
~ m_planet (a near-direct mass measurement, independent of the TTV degeneracy) and an
eccentric HARMONIC SHAPE + phase that encode the FULL e-vector, so RV pins k as well as
mass. A linear probe confirms it (R^2 on the outer planet: k2 from durations 0.16 -> RV
0.63; m2 from RV alone 0.99).

Three-arm A/B/C on the SAME systems/noise, one model spanning the resonance:
  arm A  timing only                + ratio   (the original collapse)
  arm B  timing + durations         + ratio   (h pinned, k free -> partial rescue)
  arm C  timing + durations + RV    + ratio   (h AND k pinned -> mass should snap tight)

Question: does the near-resonance (separatrix) mass tightening jump from B's ~22% toward
the ~80% seen at fixed ratio once k is constrained? And does k2's posterior finally tighten?
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # match cross_resonance.py: open a librating/chaotic zone near 2:1
S.E2_MAX = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5           # timing + duration noise (minutes)
RV_NOISE_MS = 1.0         # radial-velocity noise (m/s), HARPS-like
N_TRAIN, N_VAL, N_TEST = 13000, 1500, 3000
EPOCHS, BATCH, LR = 230, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20

# column layout of the full 150-D feature: [timing 0:60 | durations 60:120 | rv 120:150]
N_TIMEDUR = S.FEATURE_DIM_FULL                       # 120: columns that carry timing-scale noise
PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])
NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]


def gen(n, seed):
    """theta (6), per-system ratio, and the clean 150-D feature (timing+durations+RV)."""
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    ratio = rng.uniform(RATIO_LO, RATIO_HI, n)
    p2 = ratio * S.P1
    f = np.full((n, S.FEATURE_DIM_RV), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i + 2000, n))
        f[sl] = S.simulate(th[sl], p2=p2[sl], durations=True, rv=True, rv_noise_ms=0.0)
    ok = ~np.isnan(f).any(1)
    return th[ok], ratio[ok], f[ok]


def _cols(use_dur, use_rv):
    """column indices of the selected observable blocks within the 150-D feature."""
    idx = list(range(0, S.FEATURE_DIM))
    if use_dur:
        idx += list(range(S.FEATURE_DIM, S.FEATURE_DIM_FULL))
    if use_rv:
        idx += list(range(S.FEATURE_DIM_FULL, S.FEATURE_DIM_RV))
    return np.array(idx)


def train_eval(name, use_dur, use_rv, data, seed=0):
    th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te = data
    cols = _cols(use_dur, use_rv)
    rv_mask = cols >= N_TIMEDUR                         # which selected cols are RV (m/s noise)
    noise_scale = np.where(rv_mask, RV_NOISE_MS, NOISE_MIN)

    def make_X(feats, ratio):                          # [selected features | conditioning ratio]
        return np.concatenate([feats[:, cols], ratio[:, None]], axis=1)

    base = make_X(f_tr, r_tr)
    f_mean, f_std = base.mean(0), base.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    in_dim = len(cols) + 1
    rng = np.random.default_rng(99 + seed)

    def prep(feats, ratio, theta):
        sel = feats[:, cols] + rng.normal(0, 1, (len(feats), len(cols))) * noise_scale  # per-col noise
        x = (np.concatenate([sel, ratio[:, None]], 1) - f_mean) / f_std
        y = (theta - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva, Yva = prep(f_va, r_va, th_va)
    torch.manual_seed(seed)
    net = MDN(in_dim=in_dim, theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.5)
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        Xtr, Ytr = prep(f_tr, r_tr, th_tr)
        perm = torch.randperm(len(Xtr)); Xtr, Ytr = Xtr[perm], Ytr[perm]
        for i in range(0, len(Xtr), BATCH):
            opt.zero_grad(); loss = net.nll(Xtr[i:i+BATCH], Ytr[i:i+BATCH]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = net.nll(Xva, Yva).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"  [{name}] in_dim={in_dim:3d}  best val NLL {best:+.3f}  ({time.time()-t0:.0f}s)")

    # evaluate on test set (fixed noise draw for fairness)
    tn = np.random.default_rng(7)
    sel = f_te[:, cols] + tn.normal(0, 1, (len(f_te), len(cols))) * noise_scale
    x = torch.tensor((np.concatenate([sel, r_te[:, None]], 1) - f_mean) / f_std,
                     dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        samp = net.sample(x, n=1000).cpu().numpy() * th_std + th_mean   # (M,1000,6)

    dist = np.abs(r_te - 2.0)
    def stats(mask):
        idx = np.where(mask)[0]
        ratio = samp[idx].std(1).mean(0) / PRIOR_STD
        qlo, qhi = np.percentile(samp[idx], 5, 1), np.percentile(samp[idx], 95, 1)
        cov90 = np.mean((th_te[idx] >= qlo) & (th_te[idx] <= qhi), axis=0)
        return ratio, cov90.mean(), len(idx)
    return {"name": name, "val_nll": best,
            "all": stats(np.ones(len(dist), bool)),
            "sep": stats(dist < 0.025),                 # separatrix / librating
            "circ": stats(dist >= 0.05)}                # circulating


SEEDS = [0, 1, 2]
ARMS = [("timing only      ", False, False),
        ("timing + dur      ", True,  False),
        ("timing + dur + rv ", True,  True)]


def main():
    print(f"Device: {DEVICE}. 3-seed cross-resonance A/B/C (ratio U(1.9,2.2), e_max=0.15).")
    print(f"  noise: timing/dur {NOISE_MIN} min, RV {RV_NOISE_MS} m/s.")
    runs = {a[0]: [] for a in ARMS}
    for seed in SEEDS:
        t0 = time.time()
        th_tr, r_tr, f_tr = gen(N_TRAIN, seed * 10 + 0)
        th_va, r_va, f_va = gen(N_VAL, seed * 10 + 1)
        th_te, r_te, f_te = gen(N_TEST, seed * 10 + 5)
        data = (th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te)
        print(f"\nseed {seed}: usable tr {len(th_tr)}, te {len(th_te)} ({time.time()-t0:.0f}s)")
        for label, ud, ur in ARMS:
            runs[label].append(train_eval(label, ud, ur, data, seed))

    def agg(rs, key, kind):
        if kind == "nll":
            v = np.array([r["val_nll"] for r in rs]); return v.mean(), v.std()
        if kind == "ratio":
            v = np.array([r[key][0] for r in rs]); return v.mean(0), v.std(0)
        v = np.array([r[key][1] for r in rs]); return v.mean(), v.std()

    A, B, C = (runs[a[0]] for a in ARMS)
    print("\n" + "=" * 84)
    print("K-CONSTRAINT RESULT (3 seeds)  -  one model spanning ratio 1.90-2.20, e_max=0.15")
    print("  mean +/- std over seeds.  A=timing, B=+durations, C=+durations+RV.")
    print("=" * 84)
    nA, sA = agg(A, None, "nll"); nB, sB = agg(B, None, "nll"); nC, sC = agg(C, None, "nll")
    print(f"  best val NLL:  A {nA:+.2f}+/-{sA:.2f}   B {nB:+.2f}+/-{sB:.2f}   C {nC:+.2f}+/-{sC:.2f}")

    print("\n  m2 mass width ratio by distance from resonance (smaller = sharper);"
          " tightening vs A in []:")
    print(f"    {'region':>22} | {'A timing':>10} | {'B +dur':>10} | {'C +dur+rv':>10} | {'cov A/B/C':>14}")
    for key, lbl in [("sep", "separatrix |dr|<0.025"), ("circ", "circulating |dr|>0.05"),
                     ("all", "all systems")]:
        ra, _ = agg(A, key, "ratio"); rb, _ = agg(B, key, "ratio"); rc, _ = agg(C, key, "ratio")
        ca, _ = agg(A, key, "cov"); cb, _ = agg(B, key, "cov"); cc, _ = agg(C, key, "cov")
        print(f"    {lbl:>22} | {ra[1]:10.2f} | {rb[1]:5.2f}[{(1-rb[1]/ra[1])*100:3.0f}%] | "
              f"{rc[1]:5.2f}[{(1-rc[1]/ra[1])*100:3.0f}%] | {ca:.2f}/{cb:.2f}/{cc:.2f}")

    print("\n  full per-parameter width ratio (all systems), mean over seeds"
          " [tightening vs A]:")
    ra, _ = agg(A, "all", "ratio"); rb, _ = agg(B, "all", "ratio"); rc, _ = agg(C, "all", "ratio")
    print(f"    {'param':>6} | {'A timing':>10} | {'B +dur':>13} | {'C +dur+rv':>13}")
    for d in range(S.THETA_DIM):
        print(f"    {NAMES[d]:>6} | {ra[d]:10.2f} | {rb[d]:6.2f}[{(1-rb[d]/ra[d])*100:3.0f}%] | "
              f"{rc[d]:6.2f}[{(1-rc[d]/ra[d])*100:3.0f}%]")


if __name__ == "__main__":
    main()
