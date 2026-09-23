"""
Cross-code check against the incumbent (jnkepler.jaxttv, differentiable JAX N-body + HMC).

This is the stretch companion to baseline_parity.py. Where baseline_parity used our OWN
REBOUND forward model (so the comparison isolates the inference method), this runs the ACTUAL
incumbent tool: jaxttv's differentiable N-body likelihood sampled with NumPyro NUTS, on a
near-2:1 two-planet system. We infer the direct analog of our theta: two masses + two
eccentricity vectors (ecosw = our h, esinw = our k), with periods/phases fixed (as our MDN
treats them "measured"). Reports the HMC posterior widths and the per-system wall-clock, to
contrast with the amortized MDN (milliseconds) from baseline_parity.py.

MUST run in the isolated venv (jax/jnkepler need numpy>=2; our torch stack needs numpy<2):
    /tmp/jaxttv-venv/bin/python baseline_jaxttv.py
"""
import os, time, json
os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6            # solar masses
DAY = 1.0
MIN = 1.0 / 1440.0          # one minute in days
NOISE_MIN = 1.0             # transit-time precision (minutes)

# ---- truth: a near-2:1 two-planet system (analog of our theta) ----
P1, P2 = 10.0, 20.4
TIC = np.array([2.0, 5.0])
M1, M2 = 10.0 * MEARTH, 28.0 * MEARTH
ECOSW = np.array([0.05, 0.03])            # = h1, h2
ESINW = np.array([0.00, 0.04])            # = k1, k2
T_START, T_END, DT = 0.0, 500.0, P1 / 20.0

tc_lin = [TIC[i] + Pi * np.arange(int((T_END - TIC[i]) / Pi)) for i, Pi in enumerate([P1, P2])]
NPL = 2
NTR = [len(x) for x in tc_lin]
jttv0 = JaxTTV(T_START, T_END, DT, tc_lin, np.array([P1, P2]), print_info=False)

par_true = {"pmass": jnp.array([M1, M2]), "period": jnp.array([P1, P2]),
            "ecosw": jnp.array(ECOSW), "esinw": jnp.array(ESINW), "tic": jnp.array(TIC)}
tc_true = np.asarray(jttv0.get_transit_times_obs(par_true)[0])

rng = np.random.default_rng(0)
tc_obs = tc_true + rng.normal(0, NOISE_MIN * MIN, tc_true.shape)
err = np.full_like(tc_obs, NOISE_MIN * MIN)
# split flat -> per-planet list for the fitted JaxTTV object
splits = np.cumsum(NTR)[:-1]
tcobs_list = np.split(tc_obs, splits)
jttv = JaxTTV(T_START, T_END, DT, tcobs_list, np.array([P1, P2]), errorobs=np.split(err, splits),
              print_info=False)

PERIOD = jnp.array([P1, P2])
TICJ = jnp.array(TIC)
SIG = NOISE_MIN * MIN


def model():
    # priors on the analog of theta: masses (Earth->solar) + eccentricity vectors
    m = numpyro.sample("mass_Me", dist.Uniform(1.0, 60.0).expand([NPL]).to_event(1))
    ecosw = numpyro.sample("ecosw", dist.Uniform(-0.15, 0.15).expand([NPL]).to_event(1))
    esinw = numpyro.sample("esinw", dist.Uniform(-0.15, 0.15).expand([NPL]).to_event(1))
    par = {"pmass": m * MEARTH, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TICJ}
    tc = jttv.get_transit_times_obs(par)[0]
    numpyro.sample("obs", dist.Normal(tc, SIG), obs=jnp.array(tc_obs))


def main():
    print("jaxttv NUTS (differentiable N-body HMC) on a near-2:1 system.")
    print(f"  {NPL} planets, {sum(NTR)} transit times, {NOISE_MIN} min precision.")
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=300, num_samples=600, num_chains=1, progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(0))
    dt = time.time() - t0
    s = mcmc.get_samples()
    print(f"\n  jaxttv NUTS wall-clock: {dt:.0f} s  ({sum(NTR)} transit times)\n")

    truth = {"mass_Me": [M1 / MEARTH, M2 / MEARTH], "ecosw": ECOSW.tolist(), "esinw": ESINW.tolist()}
    names = [("mass_Me", "m1[Me]"), ("mass_Me", "m2[Me]"),
             ("ecosw", "h1"), ("ecosw", "h2"), ("esinw", "k1"), ("esinw", "k2")]
    print(f"  {'param':>8} {'truth':>8} {'post mean':>10} {'post std':>10}")
    out = {"nuts_seconds": round(dt, 1), "n_transits": int(sum(NTR)), "per_param": {}}
    for key, lbl in [("mass_Me", 0), ("mass_Me", 1), ("ecosw", 0), ("ecosw", 1), ("esinw", 0), ("esinw", 1)]:
        arr = np.asarray(s[key])[:, lbl]
        tval = truth[key][lbl]
        name = {"mass_Me": ["m1[Me]", "m2[Me]"], "ecosw": ["h1", "h2"], "esinw": ["k1", "k2"]}[key][lbl]
        print(f"  {name:>8} {tval:>8.3f} {arr.mean():>10.3f} {arr.std():>10.4f}")
        out["per_param"][name] = {"truth": float(tval), "mean": round(float(arr.mean()), 4),
                                  "std": round(float(arr.std()), 4)}
    json.dump(out, open("_baseline_jaxttv.json", "w"), indent=2)
    print("\n  wrote _baseline_jaxttv.json")


if __name__ == "__main__":
    main()
