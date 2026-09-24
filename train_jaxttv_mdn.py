"""
Retrain SBI on jaxttv's forward model, stage 2 (torch env): train the SAME MDN architecture on
the jaxttv-generated (theta, O-C) pairs from gen_jaxttv_data.py. Identical recipe to train.py
(0.5-min noise augmentation each epoch, standardized features/targets), just a different
simulator behind the data. Saves mdn_jaxttv.pt + norm_jaxttv.npz.

    python train_jaxttv_mdn.py
"""
import numpy as np, torch
from model import MDN

SEED = 0
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 400, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM, THETA_DIM = 60, 6

tr = np.load("jaxttv_train.npz"); va = np.load("jaxttv_val.npz")
th_tr, f_tr = tr["theta"], tr["feats"]
th_va, f_va = va["theta"], va["feats"]

torch.manual_seed(SEED)
print(f"Device: {DEVICE}. Training MDN on jaxttv-forward-model data: "
      f"{len(th_tr)} train, {len(th_va)} val.")

f_mean, f_std = f_tr.mean(0), f_tr.std(0) + 1e-8
th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
np.savez("norm_jaxttv.npz", f_mean=f_mean, f_std=f_std, th_mean=th_mean, th_std=th_std)
rng = np.random.default_rng(SEED + 99)


def prep(feats, theta):
    x = (feats + rng.normal(0, NOISE_MIN, feats.shape) - f_mean) / f_std
    y = (theta - th_mean) / th_std
    return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.float32).to(DEVICE))


Xva, Yva = prep(f_va, th_va)
net = MDN(in_dim=FEATURE_DIM, theta_dim=THETA_DIM).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.5)

best, best_state = np.inf, None
for ep in range(EPOCHS):
    net.train()
    Xtr, Ytr = prep(f_tr, th_tr)
    perm = torch.randperm(len(Xtr)); Xtr, Ytr = Xtr[perm], Ytr[perm]
    for i in range(0, len(Xtr), BATCH):
        opt.zero_grad()
        loss = net.nll(Xtr[i:i+BATCH], Ytr[i:i+BATCH]); loss.backward()
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
torch.save(net.state_dict(), "mdn_jaxttv.pt")
print(f"saved mdn_jaxttv.pt (best val NLL {best:+.3f})")
