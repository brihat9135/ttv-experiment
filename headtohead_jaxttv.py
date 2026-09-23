"""
Matched-system head-to-head, stage 2 (jaxttv venv): run jaxttv NUTS on the IDENTICAL noisy
transit times saved by headtohead_ours.py, inferring the same six quantities as our MDN
(masses + eccentricity vectors, ecosw=h, esinw=k). Period fixed to the linear-ephemeris fit;
tic sampled loosely (the ephemeris freedom our O-C detrend also has).

    /tmp/jaxttv-venv/bin/python headtohead_jaxttv.py   ->  headtohead_jaxttv.npz
"""
import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
d = np.load("headtohead_data.npz")
t1, t2 = np.asarray(d["t1_day"]), np.asarray(d["t2_day"])
P1, P2 = float(d["P1_day"]), float(d["P2_day"])
tic0 = np.array([float(d["tic1"]), float(d["tic2"])])
NOISE_MIN = float(d["noise_min"])
SIG = NOISE_MIN / 1440.0
NPL = 2
NTR = [len(t1), len(t2)]
tcobs = [t1, t2]
obs = jnp.array(np.concatenate([t1, t2]))
err = [np.full_like(t1, SIG), np.full_like(t2, SIG)]

t_start = 0.0
t_end = float(max(t1.max(), t2.max())) + 5.0
dt = P1 / 20.0
jttv = JaxTTV(t_start, t_end, dt, tcobs, np.array([P1, P2]), errorobs=err, print_info=False)

PERIOD = jnp.array([P1, P2])
TIC0 = jnp.array(tic0)


def model():
    m = numpyro.sample("mass_Me", dist.Uniform(1.0, 60.0).expand([NPL]).to_event(1))
    ecosw = numpyro.sample("ecosw", dist.Uniform(-0.15, 0.15).expand([NPL]).to_event(1))
    esinw = numpyro.sample("esinw", dist.Uniform(-0.15, 0.15).expand([NPL]).to_event(1))
    dtic = numpyro.sample("dtic", dist.Normal(0.0, 0.02).expand([NPL]).to_event(1))  # days
    par = {"pmass": m * MEARTH, "period": PERIOD, "ecosw": ecosw, "esinw": esinw,
           "tic": TIC0 + dtic}
    tc = jttv.get_transit_times_obs(par)[0]
    numpyro.sample("obs", dist.Normal(tc, SIG), obs=obs)


def main():
    print(f"jaxttv NUTS on the shared data: {NTR[0]}+{NTR[1]} transits, {NOISE_MIN} min, "
          f"P=[{P1:.3f},{P2:.3f}] d")
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=400, num_samples=800,
                num_chains=1, progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(0)); dtsec = time.time() - t0
    s = mcmc.get_samples()
    # pack into our theta order: m1,m2,h1,k1,h2,k2
    samples = np.stack([np.asarray(s["mass_Me"])[:, 0], np.asarray(s["mass_Me"])[:, 1],
                        np.asarray(s["ecosw"])[:, 0], np.asarray(s["esinw"])[:, 0],
                        np.asarray(s["ecosw"])[:, 1], np.asarray(s["esinw"])[:, 1]], axis=1)
    np.savez("headtohead_jaxttv.npz", samples=samples, nuts_seconds=dtsec)
    print(f"  jaxttv NUTS {dtsec:.0f} s  |  m2 mean/std %.2f / %.2f Me" %
          (samples[:, 1].mean(), samples[:, 1].std()))
    print("  saved headtohead_jaxttv.npz")


if __name__ == "__main__":
    main()
