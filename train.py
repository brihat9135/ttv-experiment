"""
Train the amortized posterior estimator.

Pipeline (this IS the method, in miniature):
  1. Draw many theta from the prior.
  2. Run the N-body simulator on each -> TTV feature (+ observational noise).
  3. Train the MDN to map feature -> posterior over theta, by maximizing the
     likelihood it assigns to each simulation's true theta.

After training, inference on a NEW system is a single forward pass (milliseconds)
instead of a fresh MCMC/grid search (the whole point: amortization).
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

SEED = 0
N_TRAIN = 16000
N_VAL = 1500
NOISE_MIN = 0.5          # assumed transit-timing precision (minutes)
EPOCHS = 400
BATCH = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate(n, seed):
    """Simulate n systems in chunks (vectorized N-body), return clean features+theta."""
    rng = np.random.default_rng(seed)
    theta = S.sample_prior(n, rng)
    feats = np.full((n, S.FEATURE_DIM), np.nan)
    CHUNK = 2000
    t0 = time.time()
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i+CHUNK, n))
        feats[sl] = S.simulate(theta[sl], noise_min=0.0, rng=rng)  # noise added later
        print(f"  simulated {sl.stop}/{n}  ({time.time()-t0:.1f}s)", flush=True)
    ok = ~np.isnan(feats).any(axis=1)
    return theta[ok], feats[ok]


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Generating training simulations (this runs the real N-body model)...")
    th_tr, f_tr = generate(N_TRAIN, SEED)
    th_va, f_va = generate(N_VAL, SEED+1)
    print(f"Usable: {len(th_tr)} train, {len(th_va)} val")

    rng = np.random.default_rng(SEED+99)
    # standardization stats from clean training features/targets
    f_mean, f_std = f_tr.mean(0), f_tr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    np.savez("norm.npz", f_mean=f_mean, f_std=f_std, th_mean=th_mean, th_std=th_std)
    np.savez("data.npz", th_tr=th_tr, f_tr=f_tr, th_va=th_va, f_va=f_va)

    def prep(feats, theta, noise_rng):
        x = (feats + noise_rng.normal(0, NOISE_MIN, feats.shape) - f_mean) / f_std
        y = (theta - th_mean) / th_std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    Xva, Yva = prep(f_va, th_va, rng)
    Xva, Yva = Xva.to(DEVICE), Yva.to(DEVICE)

    net = MDN(in_dim=S.FEATURE_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.5)

    best = np.inf
    for ep in range(EPOCHS):
        net.train()
        # fresh observational noise each epoch = data augmentation
        Xtr, Ytr = prep(f_tr, th_tr, rng)
        perm = torch.randperm(len(Xtr))
        Xtr, Ytr = Xtr[perm].to(DEVICE), Ytr[perm].to(DEVICE)
        tot = 0.0
        for i in range(0, len(Xtr), BATCH):
            xb, yb = Xtr[i:i+BATCH], Ytr[i:i+BATCH]
            opt.zero_grad()
            loss = net.nll(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += loss.item()*len(xb)
        sched.step()
        if ep % 20 == 0 or ep == EPOCHS-1:
            net.eval()
            with torch.no_grad():
                vnll = net.nll(Xva, Yva).item()
            if vnll < best:
                best = vnll
                torch.save(net.state_dict(), "mdn.pt")
            print(f"epoch {ep:3d}  train_nll {tot/len(Xtr):+.3f}  val_nll {vnll:+.3f}  best {best:+.3f}")

    print(f"\nDone. Best val NLL {best:.3f}. Saved model -> mdn.pt")


if __name__ == "__main__":
    main()
