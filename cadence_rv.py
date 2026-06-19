"""
Cadence sweep: how FEW radial-velocity epochs still pin k near the resonance?

observables_rv / robustness_rv used the full N_RV=30 RV epochs over the baseline. RV time
is expensive, so the practical question is how many visits you actually need. Here we
generate the full 30-epoch RV grid once, then SUBSAMPLE it to {3, 5, 8, 15, 30} evenly
spaced epochs across the same baseline (RV noise fixed at 1.0 m/s, timing/durations at
0.5 min), spanning the resonance (ratio ~ U(1.9,2.2), e_max=0.15, ratio as conditioning
input), 3 seeds. Reference is the timing+durations arm (no RV).

The mass signal lives in the RV-curve AMPLITUDE (recoverable from a handful of well-placed
epochs), but k lives in the eccentric HARMONIC SHAPE, which needs enough epochs to resolve.
So we expect mass to tighten even at low cadence while k needs more epochs -- a concrete
"how many RV visits buy the k-constraint" curve.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # match observables_rv.py
S.E2_MAX = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_NOISE = 0.5          # timing + duration noise (minutes), fixed
RV_NOISE = 1.0            # m/s on RV columns, fixed (HARPS-like)
CADENCES = [3, 5, 8, 15, 30]        # number of RV epochs kept (subsampled from N_RV=30)
SEEDS = [0, 1, 2]
N_TRAIN, N_VAL, N_TEST = 9000, 1200, 2000
EPOCHS, BATCH, LR = 200, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20

N_TIMEDUR = S.FEATURE_DIM_FULL                       # 120
RV0 = S.FEATURE_DIM_FULL                             # first RV column index
PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])


def gen(n, seed):
    """theta (6), per-system ratio, clean 150-D feature (timing+durations+30 RV epochs)."""
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


def rv_subset(n_rv):
    """evenly spaced RV-epoch column indices (within the 150-D feature), or [] for reference."""
    if n_rv is None or n_rv <= 0:
        return np.array([], dtype=int)
    idx = np.unique(np.round(np.linspace(0, S.N_RV - 1, n_rv)).astype(int))
    return RV0 + idx


def train_eval(n_rv, data, seed):
    """n_rv RV epochs kept (None = timing+dur reference). Returns (ratio[6], cov90[6], sep_ratio[6])."""
    th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te = data
    rv_cols = rv_subset(n_rv)
    cols = np.concatenate([np.arange(N_TIMEDUR), rv_cols]).astype(int)
    ncol = len(cols)
    noise_vec = np.where(cols >= RV0, RV_NOISE, TIME_NOISE)         # per-column std

    def make_X(feats, ratio):
        return np.concatenate([feats[:, cols], ratio[:, None]], axis=1)

    base = make_X(f_tr, r_tr)
    f_mean, f_std = base.mean(0), base.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(seed + 99)

    def prep(feats, ratio, theta):
        sel = feats[:, cols] + rng.normal(0, 1, (len(feats), ncol)) * noise_vec
        x = (np.concatenate([sel, ratio[:, None]], 1) - f_mean) / f_std
        y = (theta - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva, Yva = prep(f_va, r_va, th_va)
    torch.manual_seed(seed)
    net = MDN(in_dim=ncol + 1, theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.5)
    best, best_state = np.inf, None
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

    tn = np.random.default_rng(7)
    sel = f_te[:, cols] + tn.normal(0, 1, (len(f_te), ncol)) * noise_vec
    x = torch.tensor((np.concatenate([sel, r_te[:, None]], 1) - f_mean) / f_std,
                     dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        samp = net.sample(x, n=1000).cpu().numpy() * th_std + th_mean
    ratio_all = samp.std(1).mean(0) / PRIOR_STD
    qlo, qhi = np.percentile(samp, 5, 1), np.percentile(samp, 95, 1)
    cov90 = np.mean((th_te >= qlo) & (th_te <= qhi), axis=0)
    sep = np.where(np.abs(r_te - 2.0) < 0.025)[0]
    ratio_sep = samp[sep].std(1).mean(0) / PRIOR_STD
    return ratio_all, cov90, ratio_sep


def main():
    print(f"Device: {DEVICE}. RV-cadence sweep across the resonance (3 seeds).")
    print(f"  timing/dur {TIME_NOISE} min, RV {RV_NOISE} m/s; epochs kept {CADENCES} of {S.N_RV}.")
    ref = {"all": [], "cov": [], "sep": []}
    cad = {c: {"all": [], "cov": [], "sep": []} for c in CADENCES}
    for seed in SEEDS:
        t0 = time.time()
        tr = gen(N_TRAIN, seed * 10); va = gen(N_VAL, seed * 10 + 1); te = gen(N_TEST, seed * 10 + 5)
        data = (*tr, *va, *te)
        ra, ca, rs = train_eval(None, data, seed)
        ref["all"].append(ra); ref["cov"].append(ca); ref["sep"].append(rs)
        for c in CADENCES:
            ra, ca, rs = train_eval(c, data, seed)
            cad[c]["all"].append(ra); cad[c]["cov"].append(ca); cad[c]["sep"].append(rs)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    def m(d, key, idx): return np.array(d[key])[:, idx].mean()
    def sd(d, key, idx): return np.array(d[key])[:, idx].std()

    print("\n" + "=" * 80)
    print("CADENCE  -  how many RV epochs pin the near-resonance mass/k (mean +/- std, 3 seeds)")
    print("  width = posterior/prior ratio (smaller = sharper). Reference = timing+durations.")
    print("=" * 80)
    print(f"  {'RV epochs':>16} | {'m2 sep':>13} | {'m2 all':>13} | {'k2 all':>13} | {'cov(m2)':>8}")
    print(f"  {'timing+dur (ref)':>16} | {m(ref,'sep',1):.2f} +/- {sd(ref,'sep',1):.2f} | "
          f"{m(ref,'all',1):.2f} +/- {sd(ref,'all',1):.2f} | {m(ref,'all',5):.2f} +/- {sd(ref,'all',5):.2f} | "
          f"{m(ref,'cov',1):.2f}")
    for c in CADENCES:
        d = cad[c]
        print(f"  {('+RV %d epochs' % c):>16} | {m(d,'sep',1):.2f} +/- {sd(d,'sep',1):.2f} | "
              f"{m(d,'all',1):.2f} +/- {sd(d,'all',1):.2f} | {m(d,'all',5):.2f} +/- {sd(d,'all',5):.2f} | "
              f"{m(d,'cov',1):.2f}")
    rsep = m(ref, 'sep', 1); rk2 = m(ref, 'all', 5)
    print(f"\n  near-resonance m2 tightening / k2 tightening vs reference "
          f"(ref m2 sep {rsep:.2f}, k2 {rk2:.2f}):")
    for c in CADENCES:
        print(f"    {c:>2} epochs : m2 {(1 - m(cad[c],'sep',1)/rsep)*100:3.0f}%   "
              f"k2 {(1 - m(cad[c],'all',5)/rk2)*100:3.0f}%")
    print("\n  (mass sits in the RV amplitude -> survives few epochs; k sits in the harmonic"
          " shape -> needs more)")


if __name__ == "__main__":
    main()
