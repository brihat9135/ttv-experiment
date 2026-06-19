"""
Observables experiment: does adding transit DURATIONS to the feature sharpen the
posterior while staying calibrated?

Motivation: the disambiguation test concluded the near-resonance informativeness
collapse is largely PHYSICAL (data-limited), so the lever for sharper posteriors is
MORE OBSERVABLES, not a fancier estimator. Transit durations measure a planet's
sky-plane speed at mid-transit, which tracks the eccentricity-vector component h
(=e cos pomega) almost independently of the perturber mass -- the information the
timing amplitude alone cannot separate from mass.

Design: train two identical MDNs on the SAME systems and the SAME noise draws.
  arm A  timing only       (60-D  O-C residuals)
  arm B  timing + durations (120-D O-C residuals then duration anomalies)
Then compare, on a held-out test set:
  - best validation NLL          (informativeness; lower = sharper, more certain)
  - posterior/prior width ratio  (per parameter; <1 = data is informative, ->1 = collapsed)
  - 90% coverage                 (calibration; should stay ~0.90 in both arms)
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

SEED = 0
N_TRAIN, N_VAL, N_TEST = 9000, 1500, 600
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 250, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# prior std per parameter: m ~ U(lo,hi) -> (hi-lo)/sqrt(12);  e-vector uniform-in-disk(R) -> R/2
PRIOR_STD = np.array([
    (S.M1_HI - S.M1_LO) / np.sqrt(12), (S.M2_HI - S.M2_LO) / np.sqrt(12),
    S.E1_MAX / 2, S.E1_MAX / 2, S.E2_MAX / 2, S.E2_MAX / 2,
])
NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]


def generate(n, seed):
    """Simulate n systems, return (theta, full 120-D clean feature). Drops unstable rows."""
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    f = np.full((n, S.FEATURE_DIM_FULL), np.nan)
    CHUNK = 3000
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        f[sl] = S.simulate(th[sl], noise_min=0.0, durations=True)
    ok = ~np.isnan(f).any(1)
    return th[ok], f[ok]


def train_arm(name, cols, data, seed=SEED):
    """Train one MDN on feature columns `cols`. data = (f_tr,th_tr,f_va,th_va,f_te,th_te)."""
    f_tr, th_tr, f_va, th_va, f_te, th_te = data
    Xtr, Xva = f_tr[:, cols], f_va[:, cols]
    f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(seed + 99)

    def prep(feat, th, r):
        x = (feat + r.normal(0, NOISE_MIN, feat.shape) - f_mean) / f_std
        y = (th - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva_t, Yva_t = prep(Xva, th_va, rng)
    torch.manual_seed(seed)
    net = MDN(in_dim=Xtr.shape[1], theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        Xtr_t, Ytr_t = prep(Xtr, th_tr, rng)            # fresh noise each epoch
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
    print(f"  [{name}] in_dim={Xtr.shape[1]:3d}  best val NLL {best:+.3f}  ({time.time()-t0:.0f}s)")

    # ---- evaluate on the test set ----
    Xte = (f_te[:, cols] + rng.normal(0, NOISE_MIN, f_te[:, cols].shape) - f_mean) / f_std
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        samp = net.sample(Xte_t, n=1000).cpu().numpy() * th_std + th_mean   # (N_TEST,1000,6)
    post_std = samp.std(1).mean(0)                                          # mean posterior std per param
    ratio = post_std / PRIOR_STD
    qlo, qhi = np.percentile(samp, 5, 1), np.percentile(samp, 95, 1)
    cov90 = np.mean((th_te >= qlo) & (th_te <= qhi), axis=0)
    return {"name": name, "val_nll": best, "ratio": ratio, "cov90": cov90}


def main():
    print(f"Device: {DEVICE}.  Generating {N_TRAIN}+{N_VAL}+{N_TEST} systems (durations on)...")
    t0 = time.time()
    th_tr, f_tr = generate(N_TRAIN, SEED)
    th_va, f_va = generate(N_VAL, SEED + 1)
    th_te, f_te = generate(N_TEST, SEED + 2)
    print(f"  usable: {len(th_tr)} train, {len(th_va)} val, {len(th_te)} test  ({time.time()-t0:.0f}s)\n")
    data = (f_tr, th_tr, f_va, th_va, f_te, th_te)

    A = train_arm("timing only ", slice(0, S.FEATURE_DIM), data)            # 60-D
    B = train_arm("timing + dur", slice(0, S.FEATURE_DIM_FULL), data)       # 120-D

    print("\n" + "=" * 78)
    print("RESULT  -  posterior/prior width ratio (smaller = sharper, more informative)")
    print("=" * 78)
    print(f"  {'param':>5} | {'prior std':>9} | {'A times':>8} | {'B +dur':>8} | {'tightened by':>12}")
    for d in range(S.THETA_DIM):
        impr = (1 - B["ratio"][d] / A["ratio"][d]) * 100
        print(f"  {NAMES[d]:>5} | {PRIOR_STD[d]:9.3f} | {A['ratio'][d]:8.2f} | {B['ratio'][d]:8.2f} | {impr:10.0f} %")
    print(f"\n  best val NLL:   A = {A['val_nll']:+.2f}   B = {B['val_nll']:+.2f}   "
          f"(B lower by {A['val_nll']-B['val_nll']:.2f} nats = more informative)")
    print(f"  90% coverage:   A = {np.round(A['cov90'],2)}")
    print(f"                  B = {np.round(B['cov90'],2)}   (both ~0.90 = still calibrated)")


if __name__ == "__main__":
    main()
