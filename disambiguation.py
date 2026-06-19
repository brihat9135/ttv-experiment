"""
PHYSICAL vs MODEL disambiguation for the near-resonance informativeness collapse.

Across the resonance the off-the-shelf MDN stays calibrated but its posteriors balloon.
Is that because (A) the data genuinely under-constrains theta there [PHYSICAL -> the
lever is more data], or (B) the true posterior is tight/multimodal but the MDN
over-widens to stay calibrated [MODEL -> the lever is a normalizing flow]?

Method: build a likelihood-based REFERENCE posterior independent of the MDN, via ABC:
simulate a large pool from the prior at the system's KNOWN period ratio, keep the K
pool members whose simulated TTV is closest to the observed TTV. Those approximate the
true posterior. Compare REFERENCE vs MDN marginal widths, each as a fraction of the
PRIOR width, at a near-resonance ratio and a circulating control ratio.

  ref_width/prior ~ 1  -> data uninformative there (PHYSICAL)
  ref_width/prior << 1 but mdn_width/prior ~ 1 -> MDN over-widens (MODEL)
  ref ~ mdn (both small or both large) -> MDN is faithful to the true posterior
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = S.E2_MAX = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5
RATIO_LO, RATIO_HI = 1.90, 2.20
N_TRAIN, N_VAL = 12000, 1200
EPOCHS, BATCH, LR = 200, 256, 1e-3
N_POOL = 25000               # ABC prior pool per ratio
K_ABC = 150                  # accepted nearest matches
RES_RATIOS = {"near-resonance (|r-2|=0.005)": 2.005, "control circulating (|r-2|=0.10)": 2.10}


def gen(n, seed, ratio_fixed=None):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    ratio = np.full(n, ratio_fixed) if ratio_fixed else rng.uniform(RATIO_LO, RATIO_HI, n)
    feats = np.full((n, S.FEATURE_DIM), np.nan)
    for i in range(0, n, 2500):
        sl = slice(i, min(i+2500, n))
        feats[sl] = S.simulate(th[sl], p2=ratio[sl]*S.P1)
    ok = ~np.isnan(feats).any(1)
    return th[ok], ratio[ok], feats[ok]


def make_X(feats, ratio):
    return np.concatenate([feats, np.atleast_1d(ratio)[:, None]], axis=1)


# ---------------- train the conditional MDN ----------------
print("Training conditional MDN (ratio-conditioned, e_max=0.15)...")
t0 = time.time()
th_tr, r_tr, f_tr = gen(N_TRAIN, 0)
th_va, r_va, f_va = gen(N_VAL, 1)
print(f"  data in {time.time()-t0:.0f}s (usable tr {len(th_tr)})")

f_mean, f_std = make_X(f_tr, r_tr).mean(0), make_X(f_tr, r_tr).std(0)+1e-8
th_mean, th_std = th_tr.mean(0), th_tr.std(0)+1e-8     # th_std ~ PRIOR width per param
IN_DIM = S.FEATURE_DIM + 1

def prep(feats, ratio, theta, r):
    x = (make_X(feats + r.normal(0, NOISE_MIN, feats.shape), ratio) - f_mean)/f_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor((theta-th_mean)/th_std, dtype=torch.float32)

rng = np.random.default_rng(99)
Xva, Yva = prep(f_va, r_va, th_va, rng); Xva, Yva = Xva.to(DEVICE), Yva.to(DEVICE)
torch.manual_seed(0)
net = MDN(in_dim=IN_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=70, gamma=0.5)
best, best_state = np.inf, None
for ep in range(EPOCHS):
    net.train()
    Xtr, Ytr = prep(f_tr, r_tr, th_tr, rng); perm = torch.randperm(len(Xtr))
    Xtr, Ytr = Xtr[perm].to(DEVICE), Ytr[perm].to(DEVICE)
    for i in range(0, len(Xtr), BATCH):
        opt.zero_grad(); loss = net.nll(Xtr[i:i+BATCH], Ytr[i:i+BATCH]); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    sched.step()
    if ep % 40 == 0 or ep == EPOCHS-1:
        net.eval()
        with torch.no_grad(): v = net.nll(Xva, Yva).item()
        if v < best: best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
net.load_state_dict(best_state); net.eval()
print(f"  trained (best val_nll {best:+.2f})")

def mdn_std(obs, ratio):
    x = torch.tensor((make_X(obs[None], ratio) - f_mean)/f_std, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(): s = net.sample(x, n=4000).cpu().numpy()[0]*th_std + th_mean
    return s.std(0)


# ---------------- ABC reference vs MDN, per ratio ----------------
results = {}
for label, rfix in RES_RATIOS.items():
    print(f"\nBuilding ABC pool ({N_POOL}) at {label} ...")
    t0 = time.time()
    pool_th, _, pool_f = gen(N_POOL, 100, ratio_fixed=rfix)
    truth_th, _, truth_f = gen(8, 200, ratio_fixed=rfix)
    print(f"  pool+truths in {time.time()-t0:.0f}s (pool {len(pool_th)})")
    abc_frac, mdn_frac, minchi = [], [], []
    for j in range(min(4, len(truth_th))):
        obs = truth_f[j] + np.random.default_rng(300+j).normal(0, NOISE_MIN, S.FEATURE_DIM)
        chi2 = np.sum((pool_f - obs)**2, axis=1) / NOISE_MIN**2
        keep = np.argsort(chi2)[:K_ABC]
        abc_std = pool_th[keep].std(0)
        abc_frac.append(abc_std / th_std)
        mdn_frac.append(mdn_std(obs, rfix) / th_std)
        minchi.append(chi2[keep[0]])
    results[label] = (np.median(abc_frac, 0), np.median(mdn_frac, 0), np.median(minchi))

# ---------------- report ----------------
print("\n" + "="*86)
print("PHYSICAL vs MODEL — posterior width as a FRACTION OF PRIOR WIDTH (median over systems)")
print("="*86)
print("(~1.00 = as wide as the prior = uninformative;  <<1 = informative/tight)\n")
for label, (abcf, mdnf, mc) in results.items():
    print(f"{label}   [best ABC chi2 ~ {mc:.0f}, noise floor ~ {S.FEATURE_DIM}]")
    print(f"  {'param':>8} {'REFERENCE(ABC)':>16} {'MDN':>10}   verdict")
    for d in range(S.THETA_DIM):
        a, m = abcf[d], mdnf[d]
        if a > 0.7:
            v = "data uninformative (PHYSICAL)"
        elif m > 1.6*a:
            v = "MDN over-widens (MODEL)"
        else:
            v = "MDN ~ faithful"
        print(f"  {S.THETA_NAMES[d]:>8} {a:>15.2f} {m:>10.2f}   {v}")
    print()
print("-"*86)
print("Verdict logic: if the REFERENCE is already ~prior-wide near resonance, the lever is")
print("MORE DATA (longer baseline/more transits). If the REFERENCE is tight but the MDN is")
print("much wider, the lever is a BETTER MODEL (normalizing flow). Compare the two ratios.")
