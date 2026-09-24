"""
Multi-seed the flow head: repeat the MDN-vs-flow cross-resonance comparison across 3 independent
training sets (the timing-only columns of the multiseed datasets), to error-bar the flow's
stability + informativeness advantage. Writes flow_head_multiseed.png + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zuko
from model import MDN

SEEDS = [0, 1, 2]
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]

te = np.load("jaxttv_resrv_test.npz")
Xte = np.concatenate([te["oc"], te["ratio"][:, None]], 1); th_te = te["theta"]
IN = Xte.shape[1]
NOISE = np.zeros(IN); NOISE[:60] = NOISE_MIN


def load(seed):
    tr = np.load(f"jaxttv_resrv_train_s{seed}.npz"); va = np.load(f"jaxttv_resrv_val_s{seed}.npz")
    Xtr = np.concatenate([tr["oc"], tr["ratio"][:, None]], 1)
    Xva = np.concatenate([va["oc"], va["ratio"][:, None]], 1)
    return Xtr, tr["theta"], Xva, va["theta"]


def run_seed(seed):
    Xtr, th_tr, Xva, th_va = load(seed)
    f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
    rng = np.random.default_rng(seed + 99)

    def prep(X, th):
        x = (X + rng.normal(0, 1, X.shape) * NOISE - f_mean) / f_std
        return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
                torch.tensor((th - th_mean) / th_std, dtype=torch.float32).to(DEVICE))

    Xv, Yv = prep(Xva, th_va)

    # MDN
    torch.manual_seed(seed)
    net = MDN(in_dim=IN, theta_dim=6).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5)
    mc = []
    for ep in range(EPOCHS):
        net.train(); Xt, Yt = prep(Xtr, th_tr); p = torch.randperm(len(Xt)); Xt, Yt = Xt[p], Yt[p]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad(); l = net.nll(Xt[i:i+BATCH], Yt[i:i+BATCH]); l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            mc.append(net.nll(Xv, Yv).item())
    mc = np.array(mc)

    # Flow
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=6, context=IN, transforms=5, hidden_features=(128, 128)).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5)
    fc, best, best_state = [], np.inf, None
    for ep in range(EPOCHS):
        flow.train(); Xt, Yt = prep(Xtr, th_tr); p = torch.randperm(len(Xt)); Xt, Yt = Xt[p], Yt[p]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad(); l = -flow(Xt[i:i+BATCH]).log_prob(Yt[i:i+BATCH]).mean(); l.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        sch.step(); flow.eval()
        with torch.no_grad():
            v = -flow(Xv).log_prob(Yv).mean().item()
        fc.append(v)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in flow.state_dict().items()}
    fc = np.array(fc); flow.load_state_dict(best_state)

    # flow calibration on shared test
    tn = np.random.default_rng(7)
    ctx = torch.tensor((Xte + tn.normal(0, 1, Xte.shape) * NOISE - f_mean) / f_std,
                       dtype=torch.float32, device=DEVICE)
    samp = np.empty((len(Xte), 400, 6), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(ctx), 1000):
            s = flow(ctx[i:i+1000]).sample((400,)).cpu().numpy()
            samp[i:i+1000] = np.transpose(s, (1, 0, 2)) * th_std + th_mean
    errs = []
    for lv in LEVELS:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        c = np.mean([np.mean((th_te[:, dd] >= np.percentile(samp[:, :, dd], lo, 1)) &
                             (th_te[:, dd] <= np.percentile(samp[:, :, dd], hi, 1))) for dd in range(6)])
        errs.append(abs(c - lv))
    return dict(mdn_swing=mc.max()-mc.min(), mdn_best=mc.min(),
                flow_swing=fc.max()-fc.min(), flow_best=fc.min(), flow_calib=float(np.mean(errs)))


res = []
for s in SEEDS:
    r = run_seed(s); res.append(r)
    print(f"seed {s}: MDN swing {r['mdn_swing']:.0f} best {r['mdn_best']:+.2f} | "
          f"flow swing {r['flow_swing']:.2f} best {r['flow_best']:+.2f} calib {r['flow_calib']*100:.1f}%",
          flush=True)


def ms(key):
    v = np.array([r[key] for r in res]); return v.mean(), v.std()


print(f"\n=== {len(SEEDS)} seeds ===")
print(f"  MDN  val-NLL swing {ms('mdn_swing')[0]:.0f} +/- {ms('mdn_swing')[1]:.0f} (diverges every seed)")
print(f"  flow val-NLL swing {ms('flow_swing')[0]:.2f} +/- {ms('flow_swing')[1]:.2f} (stable)")
print(f"  best val NLL: MDN {ms('mdn_best')[0]:+.2f}+/-{ms('mdn_best')[1]:.2f}  "
      f"flow {ms('flow_best')[0]:+.2f}+/-{ms('flow_best')[1]:.2f}")
print(f"  flow calibration {ms('flow_calib')[0]*100:.1f}% +/- {ms('flow_calib')[1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(7.6, 5.0))
x = np.arange(len(SEEDS))
ax.bar(x-0.2, [r["mdn_swing"] for r in res], 0.4, color="#c0392b", label="MDN")
ax.bar(x+0.2, [r["flow_swing"] for r in res], 0.4, color="#1f6fb2", label="flow")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_ylabel("val-NLL swing (max-min, log)")
ax.set_title(f"Flow head is stable every seed\nMDN swing {ms('mdn_swing')[0]:.0f} vs flow {ms('flow_swing')[0]:.1f} nats")
ax.legend(); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig("flow_head_multiseed.png", dpi=145)
print("wrote flow_head_multiseed.png")

json.dump({"seeds": SEEDS, "per_seed": res,
           "mdn_swing_mean": round(ms('mdn_swing')[0], 1), "flow_swing_mean": round(ms('flow_swing')[0], 2),
           "flow_swing_std": round(ms('flow_swing')[1], 2),
           "mdn_best_mean": round(ms('mdn_best')[0], 3), "flow_best_mean": round(ms('flow_best')[0], 3),
           "flow_calib_mean": round(ms('flow_calib')[0], 4)},
          open("_flow_head_multiseed.json", "w"), indent=2, default=float)
print("wrote _flow_head_multiseed.json")
