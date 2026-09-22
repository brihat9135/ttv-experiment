"""
Robustness pass: does the RV result (observables_rv.py) survive COARSER RV precision?

observables_rv gave the RV columns an optimistic 1.0 m/s noise. Real RV precision ranges
from ~0.3 m/s (ESPRESSO) to several m/s (older spectrographs); the star's reflex signal
here is only ~6-12 m/s semi-amplitude, so at coarse precision the RV SNR approaches 1.
Here the timing + duration columns keep their 0.5-min noise (the established B arm) and
the RV columns get a separate noise swept over [0.3, 1, 3, 10] m/s, across 3 seeds, in
the same across-resonance setting (ratio ~ U(1.9,2.2), e_max=0.15, ratio as conditioning
input). The reference is the timing+durations arm (no RV).

We track, as RV precision degrades:
  - m2 mass posterior/prior width ratio at the SEPARATRIX (the headline near-resonance
    rescue: B ~16% -> C 83% tightening at 1 m/s) and over all systems,
  - k2 width ratio (the component RV uniquely pins; durations leave it free),
  - 90% coverage.
The mass + k signal lives largely in the RV-curve amplitude/shape averaged over 30 epochs,
so it should degrade gracefully and the near-resonance rescue should persist to a few m/s
before washing out near the ~10 m/s signal scale.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # match observables_rv.py / cross_resonance.py
S.E2_MAX = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_NOISE = 0.5          # timing + duration noise (minutes), fixed
RV_NOISES = [0.3, 1.0, 3.0, 10.0]   # m/s, applied to RV columns only
SEEDS = [0, 1, 2]
N_TRAIN, N_VAL, N_TEST = 9000, 1200, 2000
EPOCHS, BATCH, LR = 200, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20

N_TIMEDUR = S.FEATURE_DIM_FULL                       # 120
PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])


def gen(n, seed):
    """theta (6), per-system ratio, clean 150-D feature (timing+durations+RV)."""
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


def train_eval(use_rv, rv_noise, data, seed):
    """Returns (width_ratio[6], cov90[6], sep_ratio[6]). use_rv False = timing+dur reference."""
    th_tr, r_tr, f_tr, th_va, r_va, f_va, th_te, r_te, f_te = data
    ncol = S.FEATURE_DIM_RV if use_rv else N_TIMEDUR
    cols = np.arange(ncol)
    rvn = float(rv_noise) if (use_rv and rv_noise is not None) else 0.0
    noise_vec = np.where(cols >= N_TIMEDUR, rvn, TIME_NOISE)        # per-column std

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


def make_figure(ref, rv):
    """Two panels: near-resonance m2 tightening vs RV precision, and k2 width vs precision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean(d, key, idx): return np.array(d[key])[:, idx].mean()
    def std(d, key, idx):  return np.array(d[key])[:, idx].std()

    noises = RV_NOISES
    ref_sep = mean(ref, "sep", 1)
    ref_k2  = mean(ref, "all", 5)
    m2_tight = [(1 - mean(rv[n], "sep", 1) / ref_sep) * 100 for n in noises]
    k2_width = [mean(rv[n], "all", 5) for n in noises]
    k2_err   = [std(rv[n], "all", 5) for n in noises]
    cov_m2   = [mean(rv[n], "cov", 1) for n in noises]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.plot(noises, m2_tight, "o-", color="#1f6fb2", lw=2.2, ms=8)
    for x, y in zip(noises, m2_tight):
        ax1.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9, fontweight="bold", color="#1f6fb2")
    ax1.axhline(0, color="#c44e52", ls="--", lw=1.2, label="timing+durations (ref)")
    ax1.set_xscale("log")
    ax1.set_xticks(noises); ax1.set_xticklabels([f"{n:g}" for n in noises])
    ax1.set_xlabel("RV precision (m/s)  —  coarser →")
    ax1.set_ylabel("near-resonance m2 tightening (%)")
    ax1.set_title("Mass rescue degrades gracefully with RV noise")
    ax1.set_ylim(-8, 100); ax1.grid(alpha=0.25); ax1.legend(fontsize=8.5, loc="upper right")

    ax2.errorbar(noises, k2_width, yerr=k2_err, fmt="s-", color="#4c8c4a", lw=2.2, ms=7, capsize=3)
    ax2.axhline(ref_k2, color="#c44e52", ls="--", lw=1.2, label=f"timing+dur k2 ({ref_k2:.2f})")
    ax2.set_xscale("log")
    ax2.set_xticks(noises); ax2.set_xticklabels([f"{n:g}" for n in noises])
    ax2.set_xlabel("RV precision (m/s)  —  coarser →")
    ax2.set_ylabel("k2 posterior/prior width  (smaller = sharper)")
    ax2.set_title("k2 constraint widens back toward the reference")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8.5, loc="lower right")
    for x, c in zip(noises, cov_m2):
        ax2.annotate(f"cov {c:.2f}", (x, k2_width[noises.index(x)]),
                     textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=7.5, color="#555")

    fig.suptitle("RV-precision robustness across the 2:1 resonance (3 seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("robustness_rv.png", dpi=145)
    print("  wrote robustness_rv.png")


def dump_json(ref, rv):
    import json
    def mean(d, key, idx): return round(float(np.array(d[key])[:, idx].mean()), 4)
    ref_sep = mean(ref, "sep", 1)
    out = {"rv_noises": RV_NOISES, "ref_m2_sep": ref_sep,
           "ref_k2_all": mean(ref, "all", 5), "ref_cov_m2": mean(ref, "cov", 1),
           "per_noise": {str(n): {
               "m2_tighten_pct": round((1 - mean(rv[n], "sep", 1) / ref_sep) * 100, 1),
               "m2_sep_width": mean(rv[n], "sep", 1), "k2_all_width": mean(rv[n], "all", 5),
               "cov_m2": mean(rv[n], "cov", 1)} for n in RV_NOISES}}
    json.dump(out, open("_robustness_rv.json", "w"), indent=2)
    print("  wrote _robustness_rv.json")


def main():
    print(f"Device: {DEVICE}. RV-precision robustness across the resonance (3 seeds).")
    print(f"  timing/dur fixed {TIME_NOISE} min; RV noise swept {RV_NOISES} m/s.")
    ref = {"all": [], "cov": [], "sep": []}
    rv = {n: {"all": [], "cov": [], "sep": []} for n in RV_NOISES}
    for seed in SEEDS:
        t0 = time.time()
        tr = gen(N_TRAIN, seed * 10); va = gen(N_VAL, seed * 10 + 1); te = gen(N_TEST, seed * 10 + 5)
        data = (*tr, *va, *te)
        ra, ca, rs = train_eval(False, None, data, seed)
        ref["all"].append(ra); ref["cov"].append(ca); ref["sep"].append(rs)
        for n in RV_NOISES:
            ra, ca, rs = train_eval(True, n, data, seed)
            rv[n]["all"].append(ra); rv[n]["cov"].append(ca); rv[n]["sep"].append(rs)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    def m(d, key, idx):  # mean over seeds of param idx
        return np.array(d[key])[:, idx].mean()
    def s(d, key, idx):
        return np.array(d[key])[:, idx].std()

    print("\n" + "=" * 80)
    print("ROBUSTNESS  -  RV precision vs the near-resonance mass/k rescue (mean +/- std, 3 seeds)")
    print("  width = posterior/prior ratio (smaller = sharper). Reference = timing+durations.")
    print("=" * 80)
    print(f"  {'RV precision':>16} | {'m2 sep':>13} | {'m2 all':>13} | {'k2 all':>13} | {'cov(m2)':>8}")
    print(f"  {'timing+dur (ref)':>16} | {m(ref,'sep',1):.2f} +/- {s(ref,'sep',1):.2f} | "
          f"{m(ref,'all',1):.2f} +/- {s(ref,'all',1):.2f} | {m(ref,'all',5):.2f} +/- {s(ref,'all',5):.2f} | "
          f"{m(ref,'cov',1):.2f}")
    for n in RV_NOISES:
        d = rv[n]
        print(f"  {('+RV @ %.1f m/s' % n):>16} | {m(d,'sep',1):.2f} +/- {s(d,'sep',1):.2f} | "
              f"{m(d,'all',1):.2f} +/- {s(d,'all',1):.2f} | {m(d,'all',5):.2f} +/- {s(d,'all',5):.2f} | "
              f"{m(d,'cov',1):.2f}")
    rsep = m(ref, 'sep', 1)
    print(f"\n  near-resonance m2 tightening vs reference (timing+dur sep width {rsep:.2f}):")
    for n in RV_NOISES:
        print(f"    +RV @ {n:>4.1f} m/s : {(1 - m(rv[n],'sep',1)/rsep)*100:3.0f}%")
    print("\n  (RV uniquely pins k2; watch k2 widen back toward the reference as precision coarsens)")
    dump_json(ref, rv)
    make_figure(ref, rv)


if __name__ == "__main__":
    main()
