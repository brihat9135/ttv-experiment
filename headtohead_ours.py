"""
Matched-system head-to-head, stage 1 (our env): generate ONE system's transit times with our
REBOUND simulator, add measurement noise ONCE, and save the identical noisy times so BOTH
inference methods see the same data. Then run our amortized MDN on it.

  stage 1  headtohead_ours.py       (this file, torch env)  -> headtohead_data.npz, MDN samples
  stage 2  headtohead_jaxttv.py     (jaxttv venv)           -> jaxttv NUTS samples
  stage 3  headtohead_plot.py       (torch env)             -> headtohead.png + json

The MDN sees the O-C of the noisy times (its trained feature); jaxttv sees the noisy times
directly. Same underlying data realization, so the comparison isolates the inference method.
"""
import numpy as np, torch
import simulator as S
from model import MDN

SEED = 20260923
NOISE_MIN = 0.5                                   # transit-time precision (minutes)
DAYS_PER_YR = 365.25
THETA_TRUE = np.array([8.0, 28.0, 0.05, 0.00, 0.03, 0.04])   # m1,m2,h1,k1,h2,k2 (fixed 2.1 ratio)

# ---- raw transit times from our forward model (years -> days) ----
res = S._simulate_one((*THETA_TRUE, S.P2, False))
if res is None:
    raise RuntimeError("system was unstable; pick another THETA_TRUE")
t1_yr, t2_yr, _, _, _ = res
t1_day = np.asarray(t1_yr) * DAYS_PER_YR
t2_day = np.asarray(t2_yr) * DAYS_PER_YR

rng = np.random.default_rng(SEED)
sig_day = NOISE_MIN / 1440.0
n1 = t1_day + rng.normal(0, sig_day, t1_day.shape)     # the shared noisy data
n2 = t2_day + rng.normal(0, sig_day, t2_day.shape)

# linear ephemeris per planet (for jaxttv: fix period, guess tic)
def lin(t):
    i = np.arange(len(t)); A = np.vstack([np.ones_like(i), i]).T.astype(float)
    (t0, P), *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(t0), float(P)
t0_1, P_1 = lin(n1); t0_2, P_2 = lin(n2)

np.savez("headtohead_data.npz", theta_true=THETA_TRUE, noise_min=NOISE_MIN,
         t1_day=n1, t2_day=n2, P1_day=P_1, P2_day=P_2, tic1=t0_1, tic2=t0_2,
         theta_names=np.array(S.THETA_NAMES))
print(f"saved headtohead_data.npz: {len(n1)} inner + {len(n2)} outer transits, "
      f"P=[{P_1:.4f},{P_2:.4f}] d, noise {NOISE_MIN} min")

# ---- MDN posterior on the O-C of those same noisy times ----
oc1 = S._oc(n1 / DAYS_PER_YR, S.N_TRANSITS_1)          # minutes, from the shared noisy times
oc2 = S._oc(n2 / DAYS_PER_YR, S.N_TRANSITS_2)
feat = np.concatenate([oc1, oc2])                     # 60-D timing-only feature

norm = np.load("norm.npz")
x = (feat - norm["f_mean"]) / norm["f_std"]
net = MDN(in_dim=S.FEATURE_DIM, theta_dim=S.THETA_DIM)
net.load_state_dict(torch.load("mdn.pt", map_location="cpu")); net.eval()
with torch.no_grad():
    s = net.sample(torch.tensor(x, dtype=torch.float32).view(1, -1), n=8000).numpy()[0]
mdn = s * norm["th_std"] + norm["th_mean"]
np.savez("headtohead_mdn.npz", samples=mdn)
print(f"saved headtohead_mdn.npz: MDN posterior {mdn.shape}")
print("  m2 mean/std: %.2f / %.2f Me" % (mdn[:, 1].mean(), mdn[:, 1].std()))
