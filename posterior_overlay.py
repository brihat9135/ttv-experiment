"""
Make the before/after posterior visual: for ONE system, infer the mass-eccentricity
posterior with timing only vs timing+durations, and dump samples to _overlay.json.
Shows the degeneracy ridge (timing only) collapsing to a tight blob (timing+durations).
"""
import numpy as np, torch, json
import simulator as S
from model import MDN

SEED = 0
N_TRAIN, N_VAL = 8000, 1200
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 220, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THETA_TRUE = np.array([8.0, 28.0, 0.05, 0.00, 0.03, 0.04])   # same system as the blog plot


def generate(n, seed):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    f = S.simulate(th, noise_min=0.0, durations=True)
    ok = ~np.isnan(f).any(1)
    return th[ok], f[ok]


def train_and_sample(cols, f_tr, th_tr, f_va, th_va, obs_full):
    Xtr, Xva = f_tr[:, cols], f_va[:, cols]
    f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(SEED + 99)

    def prep(feat, th):
        x = (feat + rng.normal(0, NOISE_MIN, feat.shape) - f_mean) / f_std
        y = (th - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva_t, Yva_t = prep(Xva, th_va)
    torch.manual_seed(SEED)
    net = MDN(in_dim=Xtr.shape[1], theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
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

    x = torch.tensor(((obs_full[cols]) - f_mean) / f_std, dtype=torch.float32).to(DEVICE).view(1, -1)
    with torch.no_grad():
        s = net.sample(x, n=6000).cpu().numpy()[0] * th_std + th_mean
    return s, best


def main():
    print(f"Device: {DEVICE}. Generating + training two arms...")
    th_tr, f_tr = generate(N_TRAIN, SEED)
    th_va, f_va = generate(N_VAL, SEED + 1)

    rng = np.random.default_rng(7)
    obs_full = S.simulate(THETA_TRUE, durations=True)[0] + rng.normal(0, NOISE_MIN, S.FEATURE_DIM_FULL)

    sA, nllA = train_and_sample(slice(0, S.FEATURE_DIM), f_tr, th_tr, f_va, th_va, obs_full)
    sB, nllB = train_and_sample(slice(0, S.FEATURE_DIM_FULL), f_tr, th_tr, f_va, th_va, obs_full)
    print(f"  timing-only val NLL {nllA:+.2f} | timing+dur val NLL {nllB:+.2f}")

    def pack(s):
        e2 = np.hypot(s[:, 4], s[:, 5]); e1 = np.hypot(s[:, 2], s[:, 3])
        return {"m2": [round(float(v), 3) for v in s[:, 1]],
                "e2": [round(float(v), 4) for v in e2],
                "m1": [round(float(v), 3) for v in s[:, 0]],
                "e1": [round(float(v), 4) for v in e1]}

    out = {
        "A": pack(sA), "B": pack(sB),
        "truth": {"m1": float(THETA_TRUE[0]), "m2": float(THETA_TRUE[1]),
                  "e1": float(np.hypot(*THETA_TRUE[2:4])), "e2": float(np.hypot(*THETA_TRUE[4:6]))},
        "priors": {"m1": [S.M1_LO, S.M1_HI], "m2": [S.M2_LO, S.M2_HI],
                   "e1max": S.E1_MAX, "e2max": S.E2_MAX},
        "stats": {
            "A_m2_std": round(float(sA[:, 1].std()), 2), "B_m2_std": round(float(sB[:, 1].std()), 2),
            "A_e2_std": round(float(np.hypot(sA[:, 4], sA[:, 5]).std()), 4),
            "B_e2_std": round(float(np.hypot(sB[:, 4], sB[:, 5]).std()), 4),
        },
    }
    json.dump(out, open("_overlay.json", "w"))
    print(f"  m2 std: timing {out['stats']['A_m2_std']} -> +dur {out['stats']['B_m2_std']} Me")
    print(f"  e2 std: timing {out['stats']['A_e2_std']} -> +dur {out['stats']['B_e2_std']}")
    print("  wrote _overlay.json")


if __name__ == "__main__":
    main()
