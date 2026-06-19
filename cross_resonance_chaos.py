"""
THE CATASTROPHIC-BREAK TEST: strong chaos across the 2:1 resonance.

Same cross-resonance framework as cross_resonance.py (one model, ratio ~ U(1.9,2.2)
given as a conditioning input, inferring theta=(m1,m2,h1,k1,h2,k2)), but now in the
STRONGLY interacting regime where the off-the-shelf model is expected to truly snap:

  - giant, eccentric planets (Neptune -> Jupiter masses, e up to 0.35)
  - near 2:1 these overlap neighbouring resonances -> real chaos + dynamical INSTABILITY
  - the simulator now flags ejections/blow-ups (returns None), so instability% is real;
    unstable systems are removed from training -> the prior is TRUNCATED at the
    stability boundary (itself a failure mode for amortized inference).

Analysis: instability% and calErr binned by distance from exact resonance. We expect
instability to spike near the separatrix and calErr to cross the 'broken' threshold.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

# ---- strong-interaction regime + finer integrator step for massive planets ----
S.M1_LO, S.M1_HI = 10.0, 100.0     # M_earth (Neptune -> sub-Saturn)
S.M2_LO, S.M2_HI = 30.0, 320.0     # M_earth (Neptune -> Jupiter)
S.E1_MAX = 0.35
S.E2_MAX = 0.35
S.STEP = S.P1 / 100.0              # finer symplectic step for accuracy with giants

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5
N_TRAIN, N_VAL, N_TEST = 14000, 1500, 3000
EPOCHS, BATCH, LR = 250, 256, 1e-3
RATIO_LO, RATIO_HI = 1.90, 2.20


def gen(n, seed):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    ratio = rng.uniform(RATIO_LO, RATIO_HI, n)
    p2 = ratio * S.P1
    feats = np.full((n, S.FEATURE_DIM), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i+2000, n))
        feats[sl] = S.simulate(th[sl], p2=p2[sl])
    ok = ~np.isnan(feats).any(1)
    return th, ratio, feats, ok


print("Generating STRONG-CHAOS datasets (giant+eccentric, ratio U(1.90,2.20))...")
t0 = time.time()
th_tr, r_tr, f_tr, ok_tr = gen(N_TRAIN, 0)
th_va, r_va, f_va, ok_va = gen(N_VAL, 1)
th_te, r_te, f_te, ok_te = gen(N_TEST, 5)
print(f"  generated in {time.time()-t0:.0f}s")
print(f"  OVERALL INSTABILITY: {(1-ok_tr.mean())*100:.1f}%  (usable tr {ok_tr.sum()}/{N_TRAIN})")

th_trf, r_trf, f_trf = th_tr[ok_tr], r_tr[ok_tr], f_tr[ok_tr]
th_vaf, r_vaf, f_vaf = th_va[ok_va], r_va[ok_va], f_va[ok_va]

def make_X(feats, ratio):
    return np.concatenate([feats, ratio[:, None]], axis=1)

f_mean, f_std = make_X(f_trf, r_trf).mean(0), make_X(f_trf, r_trf).std(0) + 1e-8
th_mean, th_std = th_trf.mean(0), th_trf.std(0) + 1e-8
IN_DIM = S.FEATURE_DIM + 1

def prep(feats, ratio, theta, r):
    x = (make_X(feats + r.normal(0, NOISE_MIN, feats.shape), ratio) - f_mean) / f_std
    y = (theta - th_mean) / th_std
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

rng = np.random.default_rng(99)
Xva, Yva = prep(f_vaf, r_vaf, th_vaf, rng); Xva, Yva = Xva.to(DEVICE), Yva.to(DEVICE)

torch.manual_seed(0)
net = MDN(in_dim=IN_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=90, gamma=0.5)

print("\nTraining one model across the resonance (strong-chaos regime)...")
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

th_tef, r_tef, f_tef = th_te[ok_te], r_te[ok_te], f_te[ok_te]
fo = f_tef + np.random.default_rng(7).normal(0, NOISE_MIN, f_tef.shape)
x = torch.tensor((make_X(fo, r_tef) - f_mean)/f_std, dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    samp = net.sample(x, n=1000).cpu().numpy()*th_std + th_mean

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
print("\n" + "="*86)
print("CATASTROPHIC-BREAK RESULT — giant+eccentric, one model spanning ratio 1.90–2.20")
print("="*86)
print(f"GLOBAL: calErr {g_calerr*100:.1f}%   cov90 {g_cov90*100:.1f}%   (N={g_n}, best val_nll {best:+.2f})\n")

dist_all = np.abs(r_te - 2.0)
dist_ok = np.abs(r_tef - 2.0)
edges = [0.0, 0.01, 0.025, 0.05, 0.10, 0.151]
print(f"{'|ratio-2.0|':>14} {'instability':>12} {'N(ok)':>7} {'calErr':>8} {'cov90':>7}  regime")
print("-"*86)
for a, b in zip(edges[:-1], edges[1:]):
    sel_all = (dist_all >= a) & (dist_all < b)
    inst = np.mean(sel_all & (~ok_te[:len(dist_all)])) / max(1e-9, np.mean(sel_all))
    ce, c90, nn = calib((dist_ok >= a) & (dist_ok < b))
    regime = "separatrix/librating" if b <= 0.025 else ("near-resonant" if b <= 0.05 else "circulating")
    if ce is None:
        print(f"  [{a:.3f},{b:.3f}) {inst*100:11.1f}% {nn:>7}        --      --   {regime}")
    else:
        flag = "  <- MIS-CALIBRATED" if ce > 0.07 else ""
        print(f"  [{a:.3f},{b:.3f}) {inst*100:11.1f}% {nn:>7} {ce*100:7.1f}% {c90*100:6.1f}%  {regime}{flag}")
print("-"*86)
print("Now expect: instability% climbs toward the separatrix (prior truncation), and calErr")
print("crosses the 7% 'broken' line near resonance — the catastrophic failure of the")
print("off-the-shelf approach that a chaos-aware method must repair.")
