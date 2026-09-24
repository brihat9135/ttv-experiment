"""
Retrain SBI on jaxttv's forward model, stage 3a (jaxttv venv): generate ONE test system with
jaxttv's forward model, and run jaxttv NUTS on it. The companion torch script runs the
jaxttv-TRAINED MDN on the same data and overlays them, they should now AGREE, because both
sides share jaxttv physics (unlike headtohead.py, where the REBOUND-trained MDN disagreed).

    /tmp/jaxttv-venv/bin/python retrain_compare_jaxttv.py  ->  retrain_data.npz, retrain_jaxttv.npz
"""
import os, time; os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jnkepler.jaxttv import JaxTTV

MEARTH = 3.003e-6
P1, P2 = 10.0, 21.0
TIC = np.array([2.0, 5.0])
NIN, NOUT = 40, 20
NOISE_MIN = 0.5
SIG = NOISE_MIN / 1440.0
THETA_TRUE = np.array([8.0, 28.0, 0.05, 0.00, 0.03, 0.04])   # same system as headtohead

t1 = TIC[0] + P1 * np.arange(NIN); t2 = TIC[1] + P2 * np.arange(NOUT)
jt = JaxTTV(0.0, float(max(t1.max(), t2.max())) + 5.0, P1 / 20.0, [t1, t2],
            np.array([P1, P2]), errorobs=[np.full(NIN, SIG), np.full(NOUT, SIG)], print_info=False)
PERIOD, TICJ = jnp.array([P1, P2]), jnp.array(TIC)

par_true = {"pmass": jnp.array(THETA_TRUE[[0, 1]] * MEARTH), "period": PERIOD,
            "ecosw": jnp.array(THETA_TRUE[[2, 4]]), "esinw": jnp.array(THETA_TRUE[[3, 5]]), "tic": TICJ}
tc_true = np.asarray(jt.get_transit_times_obs(par_true)[0])
rng = np.random.default_rng(20260923)
tc_obs = tc_true + rng.normal(0, SIG, tc_true.shape)
np.savez("retrain_data.npz", tc_obs=tc_obs, theta_true=THETA_TRUE, P=[P1, P2], tic=TIC,
         nin=NIN, nout=NOUT, noise_min=NOISE_MIN)
obs = jnp.array(tc_obs)


def model():
    m = numpyro.sample("mass_Me", dist.Uniform(1.0, 60.0).expand([2]).to_event(1))
    ecosw = numpyro.sample("ecosw", dist.Uniform(-0.15, 0.15).expand([2]).to_event(1))
    esinw = numpyro.sample("esinw", dist.Uniform(-0.15, 0.15).expand([2]).to_event(1))
    dtic = numpyro.sample("dtic", dist.Normal(0.0, 0.02).expand([2]).to_event(1))
    par = {"pmass": m * MEARTH, "period": PERIOD, "ecosw": ecosw, "esinw": esinw, "tic": TICJ + dtic}
    numpyro.sample("obs", dist.Normal(jt.get_transit_times_obs(par)[0], SIG), obs=obs)


def main():
    print("jaxttv NUTS on the jaxttv-generated test system (same physics as the retrained MDN).")
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=400, num_samples=800,
                num_chains=1, progress_bar=False)
    t0 = time.time(); mcmc.run(jax.random.PRNGKey(0)); dt = time.time() - t0
    s = mcmc.get_samples()
    samples = np.stack([np.asarray(s["mass_Me"])[:, 0], np.asarray(s["mass_Me"])[:, 1],
                        np.asarray(s["ecosw"])[:, 0], np.asarray(s["esinw"])[:, 0],
                        np.asarray(s["ecosw"])[:, 1], np.asarray(s["esinw"])[:, 1]], axis=1)
    np.savez("retrain_jaxttv.npz", samples=samples, nuts_seconds=dt)
    print(f"  jaxttv NUTS {dt:.0f} s | m2 %.2f +/- %.2f Me" % (samples[:, 1].mean(), samples[:, 1].std()))
    print("  saved retrain_data.npz, retrain_jaxttv.npz")


if __name__ == "__main__":
    main()
