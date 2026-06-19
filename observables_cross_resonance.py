"""
Near-resonance version of the observables experiment: do durations rescue the
informativeness collapse in the regime where it actually bit?

cross_resonance.py showed that ONE model spanning the 2:1 resonance (ratio ~ U(1.9,2.2),
e_max=0.15, ratio as a conditioning input) stays calibrated but its posteriors COLLAPSE
toward the prior -- best val NLL fell from ~-5.9 (fixed ratio) to ~+3. Here we run the
same spanning setup as an A/B on the feature:
  arm A  timing only       + ratio   (the original collapse setup)
  arm B  timing + durations + ratio
and ask whether the extra observable restores informativeness (sharper posteriors,
lower val NLL) across the separatrix while keeping calibration.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # match cross_resonance.py: open a librating/chaotic zone near 2:1
S.E2_MAX = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5
N_TRAIN, N_VAL, N_TEST = 13000, 1500, 3000
EPOCHS, BATCH, LR = 230, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20

PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])
NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]


def gen(n, seed):
    """theta (6), per-system ratio, and 120-D clean feature (timing+durations)."""
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    ratio = rng.uniform(RATIO_LO, RATIO_HI, n)
    p2 = ratio * S.P1
    f = np.full((n, S.FEATURE_DIM_FULL), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i + 2000, n))
        f[sl] = S.simulate(th[sl], p2=p2[sl], durations=True)
    ok = ~np.isnan(f).any(1)
    return th[ok], ratio[ok], f[ok]


def train_eval(name, use_dur, data, seed=0):
    th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te = data
    cols = slice(0, S.FEATURE_DIM_FULL if use_dur else S.FEATURE_DIM)

    def make_X(feats, ratio):                      # [selected features | conditioning ratio]
        return np.concatenate([feats[:, cols], ratio[:, None]], axis=1)

    base = make_X(f_tr, r_tr)
    f_mean, f_std = base.mean(0), base.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    in_dim = (S.FEATURE_DIM_FULL if use_dur else S.FEATURE_DIM) + 1
    rng = np.random.default_rng(99 + seed)

    def prep(feats, ratio, theta):
        noisy = feats.copy()
        noisy[:, cols] = feats[:, cols] + rng.normal(0, NOISE_MIN, feats[:, cols].shape)  # noise on features, not ratio
        x = (make_X(noisy, ratio) - f_mean) / f_std
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

    # evaluate on test set
    fo = f_te.copy()
    fo[:, cols] = f_te[:, cols] + np.random.default_rng(7).normal(0, NOISE_MIN, f_te[:, cols].shape)
    x = torch.tensor((make_X(fo, r_te) - f_mean) / f_std, dtype=torch.float32).to(DEVICE)
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


def main():
    print(f"Device: {DEVICE}. 3-seed cross-resonance A/B (ratio U(1.9,2.2), e_max=0.15, durations on).")
    A_runs, B_runs = [], []
    for seed in SEEDS:
        t0 = time.time()
        th_tr, r_tr, f_tr = gen(N_TRAIN, seed * 10 + 0)
        th_va, r_va, f_va = gen(N_VAL, seed * 10 + 1)
        th_te, r_te, f_te = gen(N_TEST, seed * 10 + 5)
        data = (th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te)
        print(f"\nseed {seed}: usable tr {len(th_tr)}, te {len(th_te)} ({time.time()-t0:.0f}s)")
        A_runs.append(train_eval("timing only ", False, data, seed))
        B_runs.append(train_eval("timing + dur", True, data, seed))

    def agg(runs, key, kind):
        # kind: 'nll' scalar; 'ratio' per-param array from runs[i][key][0]; 'cov' from runs[i][key][1]
        if kind == "nll":
            v = np.array([r["val_nll"] for r in runs]); return v.mean(), v.std()
        if kind == "ratio":
            v = np.array([r[key][0] for r in runs]); return v.mean(0), v.std(0)
        v = np.array([r[key][1] for r in runs]); return v.mean(), v.std()

    nA, nA_s = agg(A_runs, None, "nll"); nB, nB_s = agg(B_runs, None, "nll")
    print("\n" + "=" * 80)
    print("NEAR-RESONANCE RESULT (3 seeds)  -  one model spanning ratio 1.90-2.20, e_max=0.15")
    print("  mean +/- std over seeds.  A = timing only, B = timing + durations.")
    print("=" * 80)
    print(f"  best val NLL:   A = {nA:+.2f} +/- {nA_s:.2f}   B = {nB:+.2f} +/- {nB_s:.2f}   "
          f"(B better by {nA-nB:.2f} nats)")

    print("\n  m2 mass width ratio by distance from resonance (smaller = sharper):")
    print(f"    {'region':>22} | {'A timing':>13} | {'B +dur':>13} | {'cov90 A/B':>11}")
    for key, lbl in [("sep", "separatrix |dr|<0.025"), ("circ", "circulating |dr|>0.05"), ("all", "all systems")]:
        ra, ras = agg(A_runs, key, "ratio"); rb, rbs = agg(B_runs, key, "ratio")
        ca, _ = agg(A_runs, key, "cov"); cb, _ = agg(B_runs, key, "cov")
        print(f"    {lbl:>22} | {ra[1]:.2f} +/- {ras[1]:.2f} | {rb[1]:.2f} +/- {rbs[1]:.2f} | {ca:.2f}/{cb:.2f}")

    print("\n  full per-parameter width ratio (all systems), mean +/- std over seeds:")
    ra, ras = agg(A_runs, "all", "ratio"); rb, rbs = agg(B_runs, "all", "ratio")
    print(f"    {'param':>6} | {'A timing':>13} | {'B +dur':>13} | {'tightened':>10}")
    for d in range(S.THETA_DIM):
        print(f"    {NAMES[d]:>6} | {ra[d]:.2f} +/- {ras[d]:.2f} | {rb[d]:.2f} +/- {rbs[d]:.2f} | {(1-rb[d]/ra[d])*100:8.0f} %")


if __name__ == "__main__":
    main()
