"""
Retrain SBI on jaxttv's forward model, stage 1 (jaxttv venv): generate (theta, O-C feature)
training pairs using jnkepler.jaxttv as the simulator instead of our REBOUND code, so the MDN
learns jaxttv's physics and becomes directly comparable to jaxttv NUTS (resolving the ~7-min
forward-model mismatch found in the head-to-head).

Same prior and feature format as our REBOUND pipeline: theta = (m1,m2,h1,k1,h2,k2), feature =
60-D timing-only O-C residuals (40 inner + 20 outer, minutes). Fixed geometry P=[10,21] d,
tic=[2,5] d (a fixed relative configuration, the jaxttv analog of our fixed PHASE1/PHASE2).
ecosw=h, esinw=k.

    /tmp/jaxttv-venv/bin/python gen_jaxttv_data.py   ->  jaxttv_train.npz, jaxttv_val.npz
"""
import os, time; os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
P1, P2 = 10.0, 21.0
TIC = np.array([2.0, 5.0])
NIN, NOUT = 40, 20
M1_LO, M1_HI, M2_LO, M2_HI, EMAX = 3.0, 15.0, 8.0, 45.0, 0.12
N_TRAIN, N_VAL = 16000, 2000
CHUNK = 2000

t1 = TIC[0] + P1 * np.arange(NIN)
t2 = TIC[1] + P2 * np.arange(NOUT)
jt = JaxTTV(0.0, float(max(t1.max(), t2.max())) + 5.0, P1 / 20.0, [t1, t2],
            np.array([P1, P2]), print_info=False)
PERIOD, TICJ = jnp.array([P1, P2]), jnp.array(TIC)


def fwd(pmass, ecosw, esinw):
    par = {"pmass": pmass, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TICJ}
    return jt.get_transit_times_obs(par)[0]
vfwd = jax.jit(jax.vmap(fwd))


def sample_prior(n, rng):
    m1 = rng.uniform(M1_LO, M1_HI, n); m2 = rng.uniform(M2_LO, M2_HI, n)
    e1 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2 * np.pi, n)
    e2 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2 * np.pi, n)
    return np.stack([m1, m2, e1 * np.cos(w1), e1 * np.sin(w1),
                     e2 * np.cos(w2), e2 * np.sin(w2)], axis=1)


def detrend_min(T):                          # (N,k) days -> O-C residual (N,k) minutes
    k = T.shape[1]; i = np.arange(k); im = i.mean(); sxx = ((i - im) ** 2).sum()
    slope = ((T - T.mean(1, keepdims=True)) * (i - im)).sum(1) / sxx
    intercept = T.mean(1) - slope * im
    return (T - (intercept[:, None] + slope[:, None] * i[None, :])) * 1440.0


def generate(n, seed):
    rng = np.random.default_rng(seed)
    th = sample_prior(n, rng)
    feats = np.full((n, NIN + NOUT), np.nan)
    t0 = time.time()
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        pmass = jnp.array(th[sl][:, [0, 1]] * MEARTH)
        ecosw = jnp.array(th[sl][:, [2, 4]])       # h1, h2
        esinw = jnp.array(th[sl][:, [3, 5]])       # k1, k2
        tc = np.asarray(vfwd(pmass, ecosw, esinw))
        feats[sl] = np.concatenate([detrend_min(tc[:, :NIN]), detrend_min(tc[:, NIN:])], axis=1)
        print(f"  {sl.stop}/{n}  ({time.time()-t0:.1f}s)", flush=True)
    ok = ~np.isnan(feats).any(1) & (np.abs(feats) < 1e4).all(1)
    return th[ok], feats[ok]


def main():
    print("Generating jaxttv-forward-model training data (theta, 60-D O-C features).")
    th_tr, f_tr = generate(N_TRAIN, 0)
    th_va, f_va = generate(N_VAL, 1)
    np.savez("jaxttv_train.npz", theta=th_tr, feats=f_tr)
    np.savez("jaxttv_val.npz", theta=th_va, feats=f_va)
    print(f"  train {f_tr.shape}, val {f_va.shape}")
    print(f"  feature O-C rms (min): inner {np.std(f_tr[:, :NIN]):.2f}, outer {np.std(f_tr[:, NIN:]):.2f}")


if __name__ == "__main__":
    main()
