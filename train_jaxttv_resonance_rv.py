"""
Break the near-resonance degeneracy in-hybrid, stage 2 (torch env): train TWO cross-resonance
MDNs on the jaxttv timing+RV data,
  arm 'timing'     : 60 O-C + ratio            (in_dim 61)
  arm 'timing_rv'  : 60 O-C + 30 RV + ratio    (in_dim 91)
Timing columns get 0.5-min noise, RV columns 1 m/s. Saves mdn_resrv_<arm>.pt + norm_resrv_<arm>.npz.
"""
import numpy as np, torch
from model import MDN

SEED = 0
NOISE_MIN, RV_NOISE = 0.5, 1.0
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOC, NRV = 60, 30

tr = np.load("jaxttv_resrv_train.npz"); va = np.load("jaxttv_resrv_val.npz")


def build_X(dset, use_rv):
    parts = [dset["oc"]]
    if use_rv:
        parts.append(dset["rv"])
    parts.append(dset["ratio"][:, None])
    return np.concatenate(parts, axis=1)


def noise_vec(ncols, use_rv):
    v = np.zeros(ncols)
    v[:NOC] = NOISE_MIN
    if use_rv:
        v[NOC:NOC + NRV] = RV_NOISE
    return v                       # ratio column stays noiseless (last)


def train(arm, use_rv):
    th_tr = tr["theta"]; X_tr = build_X(tr, use_rv)
    th_va = va["theta"]; X_va = build_X(va, use_rv)
    IN = X_tr.shape[1]
    torch.manual_seed(SEED)
    f_mean, f_std = X_tr.mean(0), X_tr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    np.savez(f"norm_resrv_{arm}.npz", f_mean=f_mean, f_std=f_std, th_mean=th_mean, th_std=th_std)
    nv = noise_vec(IN, use_rv); rng = np.random.default_rng(SEED + 99)

    def prep(X, theta):
        x = (X + rng.normal(0, 1, X.shape) * nv - f_mean) / f_std
        y = (theta - th_mean) / th_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    Xva, Yva = prep(X_va, th_va)
    net = MDN(in_dim=IN, theta_dim=6).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        Xtr, Ytr = prep(X_tr, th_tr)
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
    torch.save(net.state_dict(), f"mdn_resrv_{arm}.pt")
    print(f"  arm '{arm}' (in_dim {IN}): best val NLL {best:+.3f} -> mdn_resrv_{arm}.pt")
    return best


print(f"Device: {DEVICE}. Training two cross-resonance arms on jaxttv timing+RV data.")
b_t = train("timing", False)
b_r = train("timing_rv", True)
print(f"\nval NLL: timing {b_t:+.2f} -> timing+RV {b_r:+.2f}  (RV adds {b_t-b_r:+.2f} nats)")
