"""
Hybrid in the near-resonance regime, stage 1 (jaxttv venv): generate cross-resonance training
data with jaxttv's forward model, spanning period ratio 1.90-2.20 (the 2:1 separatrix at 2.00),
with the ratio as a conditioning input, the jaxttv analog of cross_resonance.py / observables_rv.py.

Timing-only (60-D O-C). Because the outer period (hence transit epochs) changes with ratio, we
build one JaxTTV per ratio-grid value and vmap over the orbital elements within each. Saves
theta, ratio, and the 60-D feature.

    /tmp/jaxttv-venv/bin/python gen_jaxttv_resonance.py  ->  jaxttv_res_{train,val,test}.npz
"""
import os, time; os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
P1 = 10.0
TIC = np.array([2.0, 5.0])
NIN, NOUT = 40, 20
T_END = 520.0
M1_LO, M1_HI, M2_LO, M2_HI, EMAX = 3.0, 15.0, 8.0, 45.0, 0.15   # e<=0.15, matches our near-res runs
RATIOS = np.round(np.linspace(1.90, 2.20, 21), 4)


def sample_prior(n, rng):
    m1 = rng.uniform(M1_LO, M1_HI, n); m2 = rng.uniform(M2_LO, M2_HI, n)
    e1 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2 * np.pi, n)
    e2 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2 * np.pi, n)
    return np.stack([m1, m2, e1 * np.cos(w1), e1 * np.sin(w1),
                     e2 * np.cos(w2), e2 * np.sin(w2)], axis=1)


def detrend_min(T):
    k = T.shape[1]; i = np.arange(k); im = i.mean(); sxx = ((i - im) ** 2).sum()
    slope = ((T - T.mean(1, keepdims=True)) * (i - im)).sum(1) / sxx
    intercept = T.mean(1) - slope * im
    return (T - (intercept[:, None] + slope[:, None] * i[None, :])) * 1440.0


def build_vfwd(P2):
    t1 = TIC[0] + P1 * np.arange(NIN); t2 = TIC[1] + P2 * np.arange(NOUT)
    jt = JaxTTV(0.0, T_END, P1 / 20.0, [t1, t2], np.array([P1, P2]), print_info=False)
    PERIOD, TICJ = jnp.array([P1, P2]), jnp.array(TIC)

    def fwd(pmass, ecosw, esinw):
        par = {"pmass": pmass, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TICJ}
        return jt.get_transit_times_obs(par)[0]
    return jax.jit(jax.vmap(fwd))


def generate(n_per_ratio, seed):
    rng = np.random.default_rng(seed)
    TH, RT, FE = [], [], []
    t0 = time.time()
    for r in RATIOS:
        vfwd = build_vfwd(r * P1)
        th = sample_prior(n_per_ratio, rng)
        pmass = jnp.array(th[:, [0, 1]] * MEARTH)
        ecosw = jnp.array(th[:, [2, 4]]); esinw = jnp.array(th[:, [3, 5]])
        tc = np.asarray(vfwd(pmass, ecosw, esinw))
        feat = np.concatenate([detrend_min(tc[:, :NIN]), detrend_min(tc[:, NIN:])], axis=1)
        ok = ~np.isnan(feat).any(1) & (np.abs(feat) < 1e4).all(1)
        TH.append(th[ok]); RT.append(np.full(ok.sum(), r)); FE.append(feat[ok])
    print(f"  seed {seed}: {sum(len(x) for x in TH)} systems ({time.time()-t0:.0f}s)")
    return np.concatenate(TH), np.concatenate(RT), np.concatenate(FE)


def main():
    print(f"Generating cross-resonance jaxttv data, ratio in [{RATIOS[0]},{RATIOS[-1]}] ({len(RATIOS)} grid).")
    for name, npr, seed in [("train", 600, 0), ("val", 120, 1), ("test", 200, 2)]:
        th, rt, fe = generate(npr, seed)
        np.savez(f"jaxttv_res_{name}.npz", theta=th, ratio=rt, feats=fe)
        print(f"  saved jaxttv_res_{name}.npz {fe.shape}")


if __name__ == "__main__":
    main()
