"""
Break the near-resonance degeneracy in-hybrid, stage 1 (jaxttv venv): cross-resonance jaxttv
data spanning ratio 1.90-2.20 with BOTH timing O-C (60-D) and the star's radial-velocity curve
(30 epochs, m/s, mean-subtracted). The jaxttv analog of observables_rv.py. Two arms downstream:
timing-only vs timing+RV, to show RV pins the mass and breaks the separatrix degeneracy.

    /tmp/jaxttv-venv/bin/python gen_jaxttv_resonance_rv.py  ->  jaxttv_resrv_{train,val,test}.npz
"""
import os, time; os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
P1 = 10.0
TIC = np.array([2.0, 5.0])
NIN, NOUT, NRV = 40, 20, 30
T_END = 520.0
TIMES_RV = np.linspace(1.0, 510.0, NRV)
M1_LO, M1_HI, M2_LO, M2_HI, EMAX = 3.0, 15.0, 8.0, 45.0, 0.15
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
    PERIOD, TICJ, TRV = jnp.array([P1, P2]), jnp.array(TIC), jnp.array(TIMES_RV)

    def fwd(pmass, ecosw, esinw):
        par = {"pmass": pmass, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TICJ}
        res = jt.get_transit_times_and_rvs_obs(par, TRV)   # (transit_times, rvs, energy_diag)
        return res[0], res[1]
    return jax.jit(jax.vmap(fwd))


def generate(n_per_ratio, seed):
    rng = np.random.default_rng(seed)
    TH, RT, OC, RV = [], [], [], []
    t0 = time.time()
    for r in RATIOS:
        vfwd = build_vfwd(r * P1)
        th = sample_prior(n_per_ratio, rng)
        pmass = jnp.array(th[:, [0, 1]] * MEARTH)
        ecosw = jnp.array(th[:, [2, 4]]); esinw = jnp.array(th[:, [3, 5]])
        tc, rv = vfwd(pmass, ecosw, esinw)
        tc, rv = np.asarray(tc), np.asarray(rv)
        oc = np.concatenate([detrend_min(tc[:, :NIN]), detrend_min(tc[:, NIN:])], axis=1)
        rv = rv - rv.mean(1, keepdims=True)                    # drop systemic velocity
        ok = (~np.isnan(oc).any(1) & (np.abs(oc) < 1e4).all(1)
              & ~np.isnan(rv).any(1) & (np.abs(rv) < 1e4).all(1))
        TH.append(th[ok]); RT.append(np.full(ok.sum(), r)); OC.append(oc[ok]); RV.append(rv[ok])
    print(f"  seed {seed}: {sum(len(x) for x in TH)} systems ({time.time()-t0:.0f}s)")
    return np.concatenate(TH), np.concatenate(RT), np.concatenate(OC), np.concatenate(RV)


def main():
    print(f"Cross-resonance jaxttv timing+RV data, ratio [{RATIOS[0]},{RATIOS[-1]}].")
    for name, npr, seed in [("train", 600, 0), ("val", 120, 1), ("test", 200, 2)]:
        th, rt, oc, rv = generate(npr, seed)
        np.savez(f"jaxttv_resrv_{name}.npz", theta=th, ratio=rt, oc=oc, rv=rv)
        print(f"  saved jaxttv_resrv_{name}.npz oc{oc.shape} rv{rv.shape}  "
              f"(rv rms {np.std(rv):.2f} m/s)")


if __name__ == "__main__":
    main()
