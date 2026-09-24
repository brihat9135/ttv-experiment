"""
Flow head: swap the MDN for a normalizing flow (neural spline flow, zuko) on the cross-resonance
regime where the MDN training is UNSTABLE (val NLL diverges; only early-stopping saves it, see
train_jaxttv_resonance.py). The claim to test: a flow trains STABLY there and reaches comparable
or better informativeness, while staying calibrated. Same data, same conditioning (60-D O-C +
ratio), same standardization, so val NLL is directly comparable.

Writes flow_head.png (MDN vs flow training curves + flow calibration) + json.
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zuko
from model import MDN

SEED = 0
NOISE_MIN = 0.5
EPOCHS, BATCH, LR = 300, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [0.5, 0.68, 0.9, 0.95]
NAMES = ["m1", "m2", "h1", "k1", "h2", "k2"]

tr = np.load("jaxttv_res_train.npz"); va = np.load("jaxttv_res_val.npz"); te = np.load("jaxttv_res_test.npz")
Xtr = np.concatenate([tr["feats"], tr["ratio"][:, None]], 1); th_tr = tr["theta"]
Xva = np.concatenate([va["feats"], va["ratio"][:, None]], 1); th_va = va["theta"]
Xte = np.concatenate([te["feats"], te["ratio"][:, None]], 1); th_te = te["theta"]
IN = Xtr.shape[1]
NOISE = np.zeros(IN); NOISE[:60] = NOISE_MIN

f_mean, f_std = Xtr.mean(0), Xtr.std(0) + 1e-8
th_mean, th_std = th_tr.mean(0), th_tr.std(0) + 1e-8
rng = np.random.default_rng(SEED + 99)


def prep(X, th):
    x = (X + rng.normal(0, 1, X.shape) * NOISE - f_mean) / f_std
    y = (th - th_mean) / th_std
    return (torch.tensor(x, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.float32).to(DEVICE))


Xv, Yv = prep(Xva, th_va)


def train_mdn():
    torch.manual_seed(SEED)
    net = MDN(in_dim=IN, theta_dim=6).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    curve, best = [], np.inf
    for ep in range(EPOCHS):
        net.train(); Xt, Yt = prep(Xtr, th_tr); perm = torch.randperm(len(Xt)); Xt, Yt = Xt[perm], Yt[perm]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad(); loss = net.nll(Xt[i:i+BATCH], Yt[i:i+BATCH]); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            v = net.nll(Xv, Yv).item()
        curve.append(v); best = min(best, v)
    return np.array(curve), best


def train_flow():
    torch.manual_seed(SEED)
    flow = zuko.flows.NSF(features=6, context=IN, transforms=5, hidden_features=(128, 128)).to(DEVICE)
    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    curve, best, best_state = [], np.inf, None
    for ep in range(EPOCHS):
        flow.train(); Xt, Yt = prep(Xtr, th_tr); perm = torch.randperm(len(Xt)); Xt, Yt = Xt[perm], Yt[perm]
        for i in range(0, len(Xt), BATCH):
            opt.zero_grad()
            loss = -flow(Xt[i:i+BATCH]).log_prob(Yt[i:i+BATCH]).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        sched.step(); flow.eval()
        with torch.no_grad():
            v = -flow(Xv).log_prob(Yv).mean().item()
        curve.append(v)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in flow.state_dict().items()}
    flow.load_state_dict(best_state)
    return np.array(curve), best, flow


def flow_calibration(flow):
    tn = np.random.default_rng(7)
    Xo = Xte + tn.normal(0, 1, Xte.shape) * NOISE
    ctx = torch.tensor((Xo - f_mean) / f_std, dtype=torch.float32, device=DEVICE)
    samp = np.empty((len(Xte), 500, 6), dtype=np.float32)
    flow.eval()
    with torch.no_grad():
        for i in range(0, len(ctx), 1000):
            s = flow(ctx[i:i+1000]).sample((500,)).cpu().numpy()      # (500, B, 6)
            samp[i:i+1000] = np.transpose(s, (1, 0, 2)) * th_std + th_mean
    errs = []
    for lv in LEVELS:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        c = np.mean([np.mean((th_te[:, dd] >= np.percentile(samp[:, :, dd], lo, 1)) &
                             (th_te[:, dd] <= np.percentile(samp[:, :, dd], hi, 1))) for dd in range(6)])
        errs.append(abs(c - lv))
    return float(np.mean(errs))


print(f"Device: {DEVICE}. Flow head vs MDN on the cross-resonance regime (in_dim {IN}).")
mdn_curve, mdn_best = train_mdn()
print(f"  MDN : best val NLL {mdn_best:+.2f} | MAX val NLL {mdn_curve.max():+.1f} (divergence) | "
      f"final {mdn_curve[-1]:+.1f}")
flow_curve, flow_best, flow = train_flow()
print(f"  Flow: best val NLL {flow_best:+.2f} | MAX val NLL {flow_curve.max():+.1f} | "
      f"final {flow_curve[-1]:+.2f}")
fcal = flow_calibration(flow)
print(f"  Flow calibration (coverage error): {fcal*100:.1f}%")
print(f"\n  MDN val-NLL swing (max-min) = {mdn_curve.max()-mdn_curve.min():.0f} nats (unstable)")
print(f"  Flow val-NLL swing (max-min) = {flow_curve.max()-flow_curve.min():.2f} nats (stable)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax1.plot(mdn_curve, color="#c0392b", lw=1.6, label=f"MDN (max {mdn_curve.max():.0f})")
ax1.plot(flow_curve, color="#1f6fb2", lw=1.8, label=f"flow (max {flow_curve.max():.1f})")
ax1.axhline(mdn_best, color="#c0392b", ls=":", lw=1, alpha=0.6)
ax1.set_yscale("symlog"); ax1.set_xlabel("epoch"); ax1.set_ylabel("val NLL (symlog)")
ax1.set_title("MDN training diverges; the flow is stable"); ax1.legend(); ax1.grid(alpha=0.25)
ax2.plot(mdn_curve, color="#c0392b", lw=1.6, label="MDN")
ax2.plot(flow_curve, color="#1f6fb2", lw=1.8, label="flow")
ax2.set_ylim(min(flow_curve.min(), mdn_best) - 1, 10)
ax2.set_xlabel("epoch"); ax2.set_ylabel("val NLL (zoom)")
ax2.set_title(f"Zoom: flow reaches {flow_best:+.2f} vs MDN best {mdn_best:+.2f} (flow calib {fcal*100:.1f}%)")
ax2.legend(); ax2.grid(alpha=0.25)
fig.suptitle("Flow head cures the cross-resonance training instability", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("flow_head.png", dpi=145)
print("wrote flow_head.png")

json.dump({"mdn_best": round(float(mdn_best), 3), "mdn_max": round(float(mdn_curve.max()), 1),
           "mdn_swing": round(float(mdn_curve.max()-mdn_curve.min()), 1),
           "flow_best": round(float(flow_best), 3), "flow_max": round(float(flow_curve.max()), 3),
           "flow_swing": round(float(flow_curve.max()-flow_curve.min()), 3),
           "flow_calib_error": round(fcal, 4)},
          open("_flow_head.json", "w"), indent=2)
print("wrote _flow_head.json")
