"""
Real Kepler-9 fit, stage 1 (jaxttv venv): configure jaxttv at Kepler-9's REAL geometry (periods
19.246/38.950 d, real transit ephemerides via `tic`, real observed transit numbers incl. gaps)
and generate (theta, O-C feature) training pairs. Also builds the REAL-data feature from
kepler9_ttv.csv with IDENTICAL detrending, so the flow trained here can be applied to it.

theta = (m1,m2,h1,k1,h2,k2): m1=Kepler-9b (inner, 43.4 Me published), m2=Kepler-9c (outer, 29.8).
Feature = per-transit O-C (minutes), 64 (9b) + 33 (9c) = 97-D, detrended by a linear ephemeris
over the observed transit numbers (same processing for sims and real data).

    /tmp/jaxttv-venv/bin/python kepler9_gen_jaxttv.py  ->  kepler9_train.npz
"""
import os, csv, time; os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
M_LO, M_HI, EMAX = 5.0, 80.0, 0.15          # Kepler-9-appropriate priors (covers 43.4/29.8 Me)
N_TRAIN, N_VAL = 20000, 3000
CHUNK = 2500
PUBLISHED = {"m1_Kep9b": 43.4, "m2_Kep9c": 29.8}

# ---- load real data ----
rows = {}
for r in csv.DictReader(open("kepler9_ttv.csv")):
    rows.setdefault(r["planet"], []).append((int(r["N"]), float(r["tn_BJD"]), float(r["OC_min"])))
info = {}
for p in ["Kepler-9b", "Kepler-9c"]:
    a = np.array(sorted(rows[p]))
    N = a[:, 0].astype(int); tn = a[:, 1]; oc = a[:, 2]
    A = np.vstack([np.ones_like(N), N]).T.astype(float)
    (t0, P), *_ = np.linalg.lstsq(A, tn, rcond=None)
    info[p] = dict(N=N, obs=tn + oc / 1440.0, t0=float(t0), P=float(P))
Nb, Nc = info["Kepler-9b"]["N"], info["Kepler-9c"]["N"]
nb, nc = len(Nb), len(Nc)
Pb, Pc = info["Kepler-9b"]["P"], info["Kepler-9c"]["P"]
tcobs = [info["Kepler-9b"]["obs"], info["Kepler-9c"]["obs"]]
t_end = float(max(tcobs[0].max(), tcobs[1].max())) + 20.0
jt = JaxTTV(0.0, t_end, Pb / 20.0, tcobs, np.array([Pb, Pc]),
            transit_time_method="newton", print_info=True)   # robust for large-TTV systems
PERIOD = jnp.array([Pb, Pc]); TIC = jnp.array([info["Kepler-9b"]["t0"], info["Kepler-9c"]["t0"]])
print(f"config: Pb={Pb:.4f} Pc={Pc:.4f} d (ratio {Pc/Pb:.4f}); {nb}+{nc}={nb+nc} transits")


def fwd(pmass, ecosw, esinw):
    par = {"pmass": pmass, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TIC}
    return jt.get_transit_times_obs(par)[0]
vfwd = jax.jit(jax.vmap(fwd))


def detrend(T, N):                       # T:(B,M) days, N:(M,) -> O-C (B,M) minutes
    i = N.astype(float); im = i.mean(); sxx = ((i - im) ** 2).sum()
    slope = ((T - T.mean(1, keepdims=True)) * (i - im)).sum(1) / sxx
    inter = T.mean(1) - slope * im
    return (T - (inter[:, None] + slope[:, None] * i[None, :])) * 1440.0


def feature(times):                      # times:(B, nb+nc) -> (B, nb+nc) O-C minutes
    return np.concatenate([detrend(times[:, :nb], Nb), detrend(times[:, nb:], Nc)], axis=1)


def sample_prior(n, rng):
    m1 = rng.uniform(M_LO, M_HI, n); m2 = rng.uniform(M_LO, M_HI, n)
    e1 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2 * np.pi, n)
    e2 = EMAX * np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2 * np.pi, n)
    return np.stack([m1, m2, e1*np.cos(w1), e1*np.sin(w1), e2*np.cos(w2), e2*np.sin(w2)], 1)


def generate(n, seed):
    rng = np.random.default_rng(seed); th = sample_prior(n, rng)
    feats = np.full((n, nb + nc), np.nan); t0 = time.time()
    for i in range(0, n, CHUNK):
        sl = slice(i, min(i + CHUNK, n))
        tc = np.asarray(vfwd(jnp.array(th[sl][:, [0, 1]] * MEARTH),
                             jnp.array(th[sl][:, [2, 4]]), jnp.array(th[sl][:, [3, 5]])))
        feats[sl] = feature(tc)
        print(f"  {sl.stop}/{n} ({time.time()-t0:.0f}s)", flush=True)
    ok = ~np.isnan(feats).any(1) & (np.abs(feats) < 1e5).all(1)
    return th[ok], feats[ok]


# real-data feature, same detrending
real_feat = feature(np.concatenate([info["Kepler-9b"]["obs"], info["Kepler-9c"]["obs"]])[None, :])[0]
print(f"real O-C feature: 9b rms {np.std(real_feat[:nb]):.1f} min, 9c rms {np.std(real_feat[nb:]):.1f} min")

th_tr, f_tr = generate(N_TRAIN, 0)
th_va, f_va = generate(N_VAL, 1)
np.savez("kepler9_train.npz", theta=th_tr, feats=f_tr, theta_val=th_va, feats_val=f_va,
         real_feat=real_feat, nb=nb, nc=nc, Pb=Pb, Pc=Pc,
         published=np.array([PUBLISHED["m1_Kep9b"], PUBLISHED["m2_Kep9c"]]),
         prior=np.array([M_LO, M_HI, EMAX]))
print(f"saved kepler9_train.npz: train {f_tr.shape}, val {f_va.shape}")
