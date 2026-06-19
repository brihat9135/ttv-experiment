"""
Robustness pass: does the durations result survive COARSER duration precision?

The observables_experiment gave transit durations the same 0.5-min noise as the
mid-transit times, which is optimistic -- real transit-duration uncertainties are
coarser. Here the timing columns keep 0.5-min noise but the duration columns get a
separate, larger noise, swept over [0.5, 2, 5, 10] minutes, across 3 seeds. We track
the mass posterior/prior width ratio (the headline degeneracy-breaking effect) and
90% coverage as duration precision degrades.

Note: the eccentricity signal lives mostly in the MEAN duration anomaly, so averaging
over ~40 (inner) / 20 (outer) transits suppresses per-transit duration noise by
~sqrt(N); the result should degrade gracefully rather than collapse.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

N_TRAIN, N_VAL, N_TEST = 8000, 1200, 800
TIME_NOISE = 0.5
DUR_NOISES = [0.5, 2.0, 5.0, 10.0]      # minutes, applied to duration columns only
SEEDS = [0, 1, 2]
EPOCHS, BATCH, LR = 180, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])


def generate(n, seed):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    f = np.full((n, S.FEATURE_DIM_FULL), np.nan)
    for i in range(0, n, 3000):
        sl = slice(i, min(i + 3000, n))
        f[sl] = S.simulate(th[sl], noise_min=0.0, durations=True)
    ok = ~np.isnan(f).any(1)
    return th[ok], f[ok]


def train_eval(cols, noise_vec, data, seed):
    """noise_vec: scalar or per-column std aligned with `cols`. Returns (width_ratio[6], cov90[6])."""
    f_tr, th_tr, f_va, th_va, f_te, th_te = data
    Xtr, Xva, Xte = f_tr[:, cols], f_va[:, cols], f_te[:, cols]
    f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(seed + 99)

    def prep(feat, th):
        x = (feat + rng.normal(0, 1, feat.shape) * noise_vec - f_mean) / f_std
        y = (th - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva_t, Yva_t = prep(Xva, th_va)
    torch.manual_seed(seed)
    net = MDN(in_dim=Xtr.shape[1], theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.5)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        Xtr_t, Ytr_t = prep(Xtr, th_tr)
        perm = torch.randperm(len(Xtr_t))
        Xtr_t, Ytr_t = Xtr_t[perm], Ytr_t[perm]
        for i in range(0, len(Xtr_t), BATCH):
            opt.zero_grad()
            loss = net.nll(Xtr_t[i:i+BATCH], Ytr_t[i:i+BATCH])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vnll = net.nll(Xva_t, Yva_t).item()
        if vnll < best:
            best, best_state = vnll, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    Xte_n = (f_te[:, cols] + rng.normal(0, 1, Xte.shape) * noise_vec - f_mean) / f_std
    with torch.no_grad():
        samp = net.sample(torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE), n=1000).cpu().numpy() * th_std + th_mean
    ratio = samp.std(1).mean(0) / PRIOR_STD
    qlo, qhi = np.percentile(samp, 5, 1), np.percentile(samp, 95, 1)
    cov90 = np.mean((th_te >= qlo) & (th_te <= qhi), axis=0)
    return ratio, cov90


def main():
    A_ratio, B_ratio = [], {dn: [] for dn in DUR_NOISES}
    A_cov, B_cov = [], {dn: [] for dn in DUR_NOISES}
    for seed in SEEDS:
        t0 = time.time()
        th_tr, f_tr = generate(N_TRAIN, seed * 10)
        th_va, f_va = generate(N_VAL, seed * 10 + 1)
        th_te, f_te = generate(N_TEST, seed * 10 + 2)
        data = (f_tr, th_tr, f_va, th_va, f_te, th_te)   # order matches train_eval's unpack
        r, c = train_eval(slice(0, S.FEATURE_DIM), TIME_NOISE, data, seed)
        A_ratio.append(r); A_cov.append(c)
        for dn in DUR_NOISES:
            nv = np.concatenate([np.full(S.FEATURE_DIM, TIME_NOISE), np.full(S.DURATION_DIM, dn)])
            r, c = train_eval(slice(0, S.FEATURE_DIM_FULL), nv, data, seed)
            B_ratio[dn].append(r); B_cov[dn].append(c)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    A_ratio = np.array(A_ratio); A_cov = np.array(A_cov)
    print("\n" + "=" * 76)
    print("ROBUSTNESS  -  m2 mass posterior/prior width ratio (smaller = sharper)")
    print("  mean +/- std over 3 seeds.  timing noise fixed at 0.5 min.")
    print("=" * 76)
    print(f"  {'duration noise':>16} | {'m2 ratio':>16} | {'m1 ratio':>16} | {'cov90(m2)':>10}")
    print(f"  {'timing only (A)':>16} | {A_ratio[:,1].mean():.2f} +/- {A_ratio[:,1].std():.2f}     | "
          f"{A_ratio[:,0].mean():.2f} +/- {A_ratio[:,0].std():.2f}     | {A_cov[:,1].mean():.2f}")
    for dn in DUR_NOISES:
        br = np.array(B_ratio[dn]); bc = np.array(B_cov[dn])
        print(f"  {('+dur @ %.1f min' % dn):>16} | {br[:,1].mean():.2f} +/- {br[:,1].std():.2f}     | "
              f"{br[:,0].mean():.2f} +/- {br[:,0].std():.2f}     | {bc[:,1].mean():.2f}")
    print("\n  (timing-only baseline is the no-durations reference; lower ratio under +dur = durations still help)")


if __name__ == "__main__":
    main()
