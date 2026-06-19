"""
Before/after posterior visual for the k-constraint (the RV sequel to posterior_overlay.py).

For ONE near-resonant system, infer the m2-k2 posterior with:
  arm B  timing + durations         (h pinned, k FREE  -> a ridge across k2)
  arm C  timing + durations + RV    (h AND k pinned     -> a tight blob)
The system sits on the separatrix (ratio ~ 2.0) with its eccentricity placed PURELY in
k2 (h2 = 0), the exact component durations cannot see and RV can. Renders the overlay to
posterior_overlay_rv.png and dumps the samples to _overlay_rv.json.
"""
import numpy as np, torch, json
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # near-resonance regime (match observables_rv.py)
S.E2_MAX = 0.15

SEED = 0
N_TRAIN, N_VAL = 12000, 1500
TIME_NOISE, RV_NOISE = 0.5, 1.0
EPOCHS, BATCH, LR = 220, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TIMEDUR = S.FEATURE_DIM_FULL                                    # 120

# one separatrix system; eccentricity purely in k2 (h2=0) -> the component only RV pins
THETA_TRUE = np.array([8.0, 28.0, 0.04, 0.00, 0.00, 0.11])
RATIO_TRUE = 2.00


def gen(n, seed):
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


def train_and_sample(use_rv, data, obs_full):
    th_tr, r_tr, f_tr, th_va, r_va, f_va = data
    ncol = S.FEATURE_DIM_RV if use_rv else N_TIMEDUR
    cols = np.arange(ncol)
    noise_vec = np.where(cols >= N_TIMEDUR, RV_NOISE, TIME_NOISE)

    def make_X(feats, ratio):
        return np.concatenate([feats[:, cols], ratio[:, None]], axis=1)

    base = make_X(f_tr, r_tr)
    f_mean, f_std = base.mean(0), base.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(SEED + 99)

    def prep(feats, ratio, theta):
        sel = feats[:, cols] + rng.normal(0, 1, (len(feats), ncol)) * noise_vec
        x = (np.concatenate([sel, ratio[:, None]], 1) - f_mean) / f_std
        y = (theta - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva, Yva = prep(f_va, r_va, th_va)
    torch.manual_seed(SEED)
    net = MDN(in_dim=ncol + 1, theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
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

    obs = np.concatenate([obs_full[cols], [RATIO_TRUE]])
    x = torch.tensor((obs - f_mean) / f_std, dtype=torch.float32).to(DEVICE).view(1, -1)
    with torch.no_grad():
        s = net.sample(x, n=6000).cpu().numpy()[0] * th_std + th_mean
    return s, best


def make_figure(sB, sC):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m2t, k2t = THETA_TRUE[1], THETA_TRUE[5]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(sB[:, 1], sB[:, 5], s=4, alpha=0.18, color="#c44e52",
               label="timing + durations  (k free)", rasterized=True)
    ax.scatter(sC[:, 1], sC[:, 5], s=4, alpha=0.22, color="#4c72b0",
               label="timing + durations + RV  (k pinned)", rasterized=True)
    ax.axhline(k2t, color="0.4", lw=0.8, ls=":")
    ax.axvline(m2t, color="0.4", lw=0.8, ls=":")
    ax.scatter([m2t], [k2t], marker="*", s=320, color="gold", edgecolor="k",
               linewidth=0.8, zorder=5, label="truth")
    ax.set_xlim(S.M2_LO, S.M2_HI)
    ax.set_ylim(-S.E2_MAX, S.E2_MAX)
    ax.set_xlabel(r"$m_2$  [$M_\oplus$]")
    ax.set_ylabel(r"$k_2 = e_2 \sin\varpi_2$")
    ax.set_title("Near-resonant posterior: RV pins the mass and $k$\n"
                 f"($m_2$ std {sB[:,1].std():.1f}→{sC[:,1].std():.1f} $M_\\oplus$, "
                 f"$k_2$ std {sB[:,5].std():.3f}→{sC[:,5].std():.3f})", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, markerscale=2)
    fig.tight_layout()
    fig.savefig("posterior_overlay_rv.png", dpi=140)
    print("  wrote posterior_overlay_rv.png")


def main():
    print(f"Device: {DEVICE}. Generating + training two arms (near-resonance)...")
    th_tr, r_tr, f_tr = gen(N_TRAIN, SEED)
    th_va, r_va, f_va = gen(N_VAL, SEED + 1)
    data = (th_tr, r_tr, f_tr, th_va, r_va, f_va)

    rng = np.random.default_rng(7)
    obs_full = S.simulate(THETA_TRUE, p2=np.array([RATIO_TRUE * S.P1]),
                          durations=True, rv=True, rv_noise_ms=0.0)[0]
    noise = np.where(np.arange(S.FEATURE_DIM_RV) >= N_TIMEDUR, RV_NOISE, TIME_NOISE)
    obs_full = obs_full + rng.normal(0, 1, S.FEATURE_DIM_RV) * noise

    sB, nllB = train_and_sample(False, data, obs_full)
    sC, nllC = train_and_sample(True, data, obs_full)
    print(f"  timing+dur val NLL {nllB:+.2f} | timing+dur+RV val NLL {nllC:+.2f}")
    print(f"  m2 std: +dur {sB[:,1].std():.2f} -> +dur+RV {sC[:,1].std():.2f} Me")
    print(f"  k2 std: +dur {sB[:,5].std():.4f} -> +dur+RV {sC[:,5].std():.4f}")

    def pack(s):
        return {"m2": [round(float(v), 3) for v in s[:, 1]],
                "k2": [round(float(v), 4) for v in s[:, 5]],
                "h2": [round(float(v), 4) for v in s[:, 4]]}
    out = {"B": pack(sB), "C": pack(sC),
           "truth": {"m2": float(THETA_TRUE[1]), "k2": float(THETA_TRUE[5]),
                     "h2": float(THETA_TRUE[4]), "ratio": RATIO_TRUE},
           "priors": {"m2": [S.M2_LO, S.M2_HI], "k2max": S.E2_MAX},
           "stats": {"B_m2_std": round(float(sB[:,1].std()), 2),
                     "C_m2_std": round(float(sC[:,1].std()), 2),
                     "B_k2_std": round(float(sB[:,5].std()), 4),
                     "C_k2_std": round(float(sC[:,5].std()), 4)}}
    json.dump(out, open("_overlay_rv.json", "w"))
    print("  wrote _overlay_rv.json")
    make_figure(sB, sC)


if __name__ == "__main__":
    main()
