"""
Follow-up: the e<=0.12 sweep did NOT break calibration (it degraded gracefully).
Hypothesis for the real break: HIGHER eccentricity. Near 2:1, larger e drives
resonance overlap -> strong chaos + dynamical instability. Unstable systems get
filtered from training (truncating the prior near the stability boundary), and the
surviving chaotic systems have a rugged theta->TTV map whose posterior fractures.

Same fixed off-the-shelf recipe; only difference vs resonance_sweep.py is e_max=0.30.
"""
import time, numpy as np, torch
import simulator as S
from model import MDN

# ---- the only change from the first sweep: eccentric planets ----
S.E1_MAX = 0.30
S.E2_MAX = 0.30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_MIN = 0.5
N_TRAIN, N_VAL, N_TEST = 7000, 800, 1000
EPOCHS, BATCH, LR = 150, 256, 1e-3
P2_DAYS = [21.0, 20.5, 20.3, 20.15, 20.05]


def gen(n, seed):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    f = np.full((n, S.FEATURE_DIM), np.nan)
    for i in range(0, n, 2000):
        sl = slice(i, min(i+2000, n))
        f[sl] = S.simulate(th[sl])
    ok = ~np.isnan(f).any(1)
    return th, f, ok


print(f"Phase 1: generating datasets (e_max={S.E1_MAX}) across the resonance approach...")
data = {}
for P2 in P2_DAYS:
    S.P2 = P2 * S.DAY
    ratio = P2 / (S.P1 / S.DAY)
    superP = 1.0/abs(2/(P2*S.DAY) - 1/S.P1) / S.DAY
    t0 = time.time()
    th_tr, f_tr, ok_tr = gen(N_TRAIN, 0)
    th_va, f_va, ok_va = gen(N_VAL, 1)
    th_te, f_te, ok_te = gen(N_TEST, 5)
    fail = 1 - ok_tr.mean()
    ttv_rms = float(np.nanmedian(np.nanstd(f_tr[ok_tr][:, :S.N_TRANSITS_1], 1)))
    data[P2] = dict(ratio=ratio, superP=superP, fail=fail, ttv_rms=ttv_rms,
                    tr=(th_tr[ok_tr], f_tr[ok_tr]), va=(th_va[ok_va], f_va[ok_va]), te=(th_te[ok_te], f_te[ok_te]))
    print(f"  ratio {ratio:.3f}  superP={superP:6.0f}d  unstable={fail*100:4.1f}%  "
          f"TTVrms={ttv_rms:5.1f}min  usable_tr={ok_tr.sum()}  ({time.time()-t0:.0f}s)")


def train_eval(d, seed=0):
    th_tr, f_tr = d["tr"]; th_va, f_va = d["va"]; th_te, f_te = d["te"]
    fm, fs = f_tr.mean(0), f_tr.std(0)+1e-8
    tm, ts = th_tr.mean(0), th_tr.std(0)+1e-8
    rng = np.random.default_rng(seed+99)
    def prep(f, th, r):
        return (torch.tensor((f + r.normal(0, NOISE_MIN, f.shape) - fm)/fs, dtype=torch.float32),
                torch.tensor((th - tm)/ts, dtype=torch.float32))
    Xva, Yva = prep(f_va, th_va, rng); Xva, Yva = Xva.to(DEVICE), Yva.to(DEVICE)
    torch.manual_seed(seed)
    net = MDN(in_dim=S.FEATURE_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        net.train()
        Xtr, Ytr = prep(f_tr, th_tr, rng); perm = torch.randperm(len(Xtr))
        Xtr, Ytr = Xtr[perm].to(DEVICE), Ytr[perm].to(DEVICE)
        for i in range(0, len(Xtr), BATCH):
            opt.zero_grad(); loss = net.nll(Xtr[i:i+BATCH], Ytr[i:i+BATCH]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step()
        if ep % 15 == 0 or ep == EPOCHS-1:
            net.eval()
            with torch.no_grad(): v = net.nll(Xva, Yva).item()
            if v < best: best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    fo = f_te + np.random.default_rng(seed+7).normal(0, NOISE_MIN, f_te.shape)
    x = torch.tensor((fo - fm)/fs, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(): samp = net.sample(x, n=1000).cpu().numpy()*ts + tm
    covs, calerr = {}, []
    for lv in [0.5, 0.68, 0.9]:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        for dd in range(S.THETA_DIM):
            ql = np.percentile(samp[:, :, dd], lo, axis=1); qh = np.percentile(samp[:, :, dd], hi, axis=1)
            c = np.mean((th_te[:, dd] >= ql) & (th_te[:, dd] <= qh)); calerr.append(abs(c-lv)); covs.setdefault(lv, []).append(c)
    return dict(cov50=np.mean(covs[0.5]), cov90=np.mean(covs[0.9]), calerr=float(np.mean(calerr)),
                m2w=float(np.median(samp[:, :, 1].std(1))), m2err=float(np.sqrt(np.mean((samp[:, :, 1].mean(1)-th_te[:, 1])**2))))


print("\nPhase 2: training one off-the-shelf model per resonance distance...")
rows = []
for P2 in P2_DAYS:
    t0 = time.time(); r = train_eval(data[P2]); rows.append((P2, data[P2], r))
    print(f"  ratio {data[P2]['ratio']:.3f} done  cov90={r['cov90']*100:4.1f}%  calErr={r['calerr']*100:4.1f}%  ({time.time()-t0:.0f}s)")

print("\n" + "="*94)
print(f"WHERE IT BREAKS — ECCENTRIC sweep (e_max={S.E1_MAX}, recipe fixed)")
print("="*94)
print(f"{'ratio':>6} {'superP/base':>11} {'unstable%':>9} {'TTVrms':>7} {'cov50':>6} {'cov90':>6} {'calErr':>7} {'m2 sigma':>9} {'m2 RMSE':>8}")
print("-"*94)
base = S.BASELINE/S.DAY
for P2, d, r in rows:
    flag = "  <- MIS-CALIBRATED" if r['calerr'] > 0.07 else ""
    print(f"{d['ratio']:6.3f} {d['superP']/base:11.1f} {d['fail']*100:8.1f}% {d['ttv_rms']:7.1f} "
          f"{r['cov50']*100:5.1f}% {r['cov90']*100:5.1f}% {r['calerr']*100:6.1f}% {r['m2w']:9.1f} {r['m2err']:8.1f}{flag}")
print("-"*94)
print("calErr > ~7% = honest-error-bar property has broken. unstable% = chaos/ejection rate;")
print("a high unstable% also means the training prior is truncated at the stability boundary.")
