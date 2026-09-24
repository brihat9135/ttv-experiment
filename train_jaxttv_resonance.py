"""
Hybrid in the near-resonance regime, stage 2 (torch env): train the MDN on the cross-resonance
jaxttv data, with the period ratio as a conditioning input (in_dim = 60 O-C + 1 ratio = 61).
Saves mdn_jaxttv_res.pt + norm_jaxttv_res.npz.
"""
import numpy as np, torch
from model import MDN

SEED = 0
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tr = np.load("jaxttv_res_train.npz"); va = np.load("jaxttv_res_val.npz")
th_tr = tr["theta"]; X_tr = np.concatenate([tr["feats"], tr["ratio"][:, None]], axis=1)
th_va = va["theta"]; X_va = np.concatenate([va["feats"], va["ratio"][:, None]], axis=1)
IN_DIM = X_tr.shape[1]

torch.manual_seed(SEED)
print(f"Device: {DEVICE}. Cross-resonance hybrid MDN: {len(th_tr)} train, in_dim {IN_DIM}.")

f_mean, f_std = X_tr.mean(0), X_tr.std(0) + 1e-8
th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
np.savez("norm_jaxttv_res.npz", f_mean=f_mean, f_std=f_std, th_mean=th_mean, th_std=th_std)
rng = np.random.default_rng(SEED + 99)
NOISE_COLS = X_tr.shape[1] - 1     # add timing noise to O-C columns, not the ratio column


def prep(X, theta):
    noise = np.zeros_like(X); noise[:, :NOISE_COLS] = rng.normal(0, NOISE_MIN, (len(X), NOISE_COLS))
    x = (X + noise - f_mean) / f_std
    y = (theta - th_mean) / th_std
    return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.float32).to(DEVICE))


Xva, Yva = prep(X_va, th_va)
net = MDN(in_dim=IN_DIM, theta_dim=6).to(DEVICE)
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
    if ep % 40 == 0 or ep == EPOCHS - 1:
        print(f"  epoch {ep:3d}  val NLL {v:+.3f}  (best {best:+.3f})", flush=True)
net.load_state_dict(best_state)
torch.save(net.state_dict(), "mdn_jaxttv_res.pt")
print(f"saved mdn_jaxttv_res.pt (best val NLL {best:+.3f})")
