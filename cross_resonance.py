"""
THE DECISIVE TEST: can ONE off-the-shelf amortized model span the 2:1 resonance?

Unlike resonance_sweep*.py (one model per fixed period ratio), here a SINGLE model is
trained over a prior with the period ratio P2/P1 ~ U(1.9, 2.2) — circulating below the
resonance, librating inside it, and circulating above, crossing the chaotic separatrix.
The measured ratio is given to the network as a CONDITIONING input (periods are
measured, not inferred); the network still infers theta = (m1,m2,h1,k1,h2,k2).

This is the survey-realistic setting: for a real system you do not know a priori which
side of resonance it sits on. The key analysis is coverage BINNED BY DISTANCE FROM
EXACT RESONANCE |ratio - 2.0| — if the off-the-shelf approach breaks, calibration error
spikes in the bin straddling the separatrix.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

S.E1_MAX = 0.15            # enough eccentricity to open a librating/chaotic zone near 2:1
S.E2_MAX = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5
N_TRAIN, N_VAL, N_TEST = 16000, 1500, 3000
EPOCHS, BATCH, LR = 250, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20
P1_DAYS = S.P1 / S.DAY


def gen(n, seed):
    """Sample theta (6) + a per-system ratio; simulate with per-system P2."""
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    ratio = rng.uniform(RATIO_LO, RATIO_HI, n)
    p2 = ratio * S.P1                       # years
    feats = np.full((n, S.FEATURE_DIM), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i+2000, n))
        feats[sl] = S.simulate(th[sl], p2=p2[sl])
    ok = ~np.isnan(feats).any(1)
    return th, ratio, feats, ok


print("Generating cross-resonance datasets (ratio ~ U(1.90, 2.20), e_max=0.15)...")
t0 = time.time()
th_tr, r_tr, f_tr, ok_tr = gen(N_TRAIN, 0)
th_va, r_va, f_va, ok_va = gen(N_VAL, 1)
th_te, r_te, f_te, ok_te = gen(N_TEST, 5)
print(f"  generated in {time.time()-t0:.0f}s | usable: tr {ok_tr.sum()}/{N_TRAIN}, "
      f"va {ok_va.sum()}/{N_VAL}, te {ok_te.sum()}/{N_TEST}")
print(f"  overall instability (failed sims): {(1-ok_tr.mean())*100:.1f}%")

# keep raw test (with ok mask + ratio) for per-bin instability before filtering
th_trf, r_trf, f_trf = th_tr[ok_tr], r_tr[ok_tr], f_tr[ok_tr]
th_vaf, r_vaf, f_vaf = th_va[ok_va], r_va[ok_va], f_va[ok_va]

# ----- build inputs: [TTV features | conditioning ratio], standardized -----
def make_X(feats, ratio):
    return np.concatenate([feats, ratio[:, None]], axis=1)

Xtr_raw = make_X(f_trf, r_trf)
f_mean, f_std = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-8
th_mean, th_std = th_trf.mean(0), th_trf.std(0) + 1e-8
IN_DIM = S.FEATURE_DIM + 1


def prep(feats, ratio, theta, r):
    x = make_X(feats + r.normal(0, NOISE_MIN, feats.shape), ratio)   # noise on TTVs only
    x = (x - f_mean) / f_std
    y = (theta - th_mean) / th_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


rng = np.random.default_rng(99)
Xva, Yva = prep(f_vaf, r_vaf, th_vaf, rng); Xva, Yva = Xva.to(DEVICE), Yva.to(DEVICE)

torch.manual_seed(0)
net = MDN(in_dim=IN_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.5)

print("\nTraining one model across the resonance...")
best, best_state = np.inf, None
for ep in range(EPOCHS):
    net.train()
    Xtr, Ytr = prep(f_trf, r_trf, th_trf, rng)
    perm = torch.randperm(len(Xtr)); Xtr, Ytr = Xtr[perm].to(DEVICE), Ytr[perm].to(DEVICE)
    for i in range(0, len(Xtr), BATCH):
        opt.zero_grad(); loss = net.nll(Xtr[i:i+BATCH], Ytr[i:i+BATCH]); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    sched.step()
    if ep % 30 == 0 or ep == EPOCHS-1:
        net.eval()
        with torch.no_grad(): v = net.nll(Xva, Yva).item()
        if v < best: best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        print(f"  epoch {ep:3d}  val_nll {v:+.3f}  best {best:+.3f}")
net.load_state_dict(best_state); net.eval()

# ----- evaluate: global + per-(distance-from-resonance) coverage -----
th_tef, r_tef, f_tef = th_te[ok_te], r_te[ok_te], f_te[ok_te]
fo = f_tef + np.random.default_rng(7).normal(0, NOISE_MIN, f_tef.shape)
x = torch.tensor((make_X(fo, r_tef) - f_mean)/f_std, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    samp = net.sample(x, n=1000).cpu().numpy()*th_std + th_mean        # (M,1000,6)

def calib(mask):
    idx = np.where(mask)[0]
    if len(idx) < 40: return None, None, len(idx)
    cerr, cov90 = [], []
    for lv in [0.5, 0.68, 0.9]:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        for d in range(S.THETA_DIM):
            ql = np.percentile(samp[idx, :, d], lo, axis=1); qh = np.percentile(samp[idx, :, d], hi, axis=1)
            c = np.mean((th_tef[idx, d] >= ql) & (th_tef[idx, d] <= qh))
            cerr.append(abs(c-lv))
            if lv == 0.9: cov90.append(c)
    return float(np.mean(cerr)), float(np.mean(cov90)), len(idx)

g_calerr, g_cov90, g_n = calib(np.ones(len(th_tef), bool))
print("\n" + "="*82)
print("CROSS-RESONANCE RESULT — one model spanning ratio 1.90–2.20 (e_max=0.15)")
print("="*82)
print(f"GLOBAL: calErr {g_calerr*100:.1f}%   cov90 {g_cov90*100:.1f}%   (N={g_n})\n")

# per-bin: distance from exact resonance, on the FULL (pre-filter) test set for instability
dist_all = np.abs(r_te - 2.0)
dist_ok = np.abs(r_tef - 2.0)
edges = [0.0, 0.01, 0.025, 0.05, 0.10, 0.151]
print(f"{'|ratio-2.0|':>14} {'instability':>12} {'N(ok)':>7} {'calErr':>8} {'cov90':>7}  regime")
print("-"*82)
for a, b in zip(edges[:-1], edges[1:]):
    inst = np.mean((dist_all >= a) & (dist_all < b) &
                   (~ok_te[:len(dist_all)])) / max(1e-9, np.mean((dist_all >= a) & (dist_all < b)))
    mask = (dist_ok >= a) & (dist_ok < b)
    ce, c90, nn = calib(mask)
    regime = "separatrix/librating" if b <= 0.025 else ("near-resonant" if b <= 0.05 else "circulating")
    if ce is None:
        print(f"  [{a:.3f},{b:.3f}) {inst*100:11.1f}% {nn:>7}        --      --   {regime}")
    else:
        flag = "  <- MIS-CALIBRATED" if ce > 0.07 else ""
        print(f"  [{a:.3f},{b:.3f}) {inst*100:11.1f}% {nn:>7} {ce*100:7.1f}% {c90*100:6.1f}%  {regime}{flag}")
print("-"*82)
print("If calErr spikes (>~7%) in the innermost bins, the off-the-shelf model breaks at the")
print("separatrix — the precise, publishable failure mode that motivates a chaos-aware method.")
print("If it stays flat, amortized SBI is robust even across the resonance (also a real result).")
