"""
Baseline parity (the "gold-standard check" the roadmap calls for, and the honest form of
the jaxttv comparison).

Question: is our fast AMORTIZED posterior (one forward pass of the MDN) as tight and as
accurate as doing inference the proper Bayesian way, per-system MCMC over the SAME forward
model? This is exactly what an HMC code like jaxttv would give you, except we use our own
REBOUND forward model as the likelihood so the comparison is apples-to-apples (same physics,
same data, same noise, same priors) and not confounded by a different N-body engine.

Setup matches train.py / mdn.pt: timing-only 60-D O-C residuals, 0.5-min noise, fixed 2:1
ratio, priors from S.sample_prior. For one held-out system we:
  1. simulate its noisy transit-timing data,
  2. get the MDN posterior (amortized, milliseconds),
  3. get the MCMC posterior (emcee over the REBOUND likelihood, the gold standard),
  4. compare per-parameter mean/width and overlay the marginals.

If the amortized posterior matches the MCMC posterior (widths ~equal, both bracket truth),
the network has learned the true posterior, and buys a ~10^4x speed-up per system on top.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"      # each worker single-threaded; emcee handles parallelism
import time, json, numpy as np, torch
from multiprocessing import Pool
import emcee
import simulator as S
from model import MDN

S._MAXPROC = 1                            # avoid nested pools; emcee owns the parallelism
DEVICE = "cpu"                            # inference is one tiny forward pass
NOISE_MIN = 0.5
SEED = 20260922

# one system with real (mass-eccentricity) structure, fixed 2:1 like the trained model
THETA_TRUE = np.array([8.0, 28.0, 0.05, 0.00, 0.03, 0.04])   # m1,m2,h1,k1,h2,k2
NWALKERS, NSTEPS, NBURN = 64, 2500, 800

norm = np.load("norm.npz")
F_MEAN, F_STD = norm["f_mean"], norm["f_std"]
TH_MEAN, TH_STD = norm["th_mean"], norm["th_std"]

rng = np.random.default_rng(SEED)
_clean = S.simulate(THETA_TRUE)[0]
OBS = _clean + rng.normal(0, NOISE_MIN, _clean.shape)        # the fixed observed data


# ---------- amortized posterior (MDN) ----------
def mdn_posterior(n=8000):
    net = MDN(in_dim=S.FEATURE_DIM, theta_dim=S.THETA_DIM).to(DEVICE)
    net.load_state_dict(torch.load("mdn.pt", map_location=DEVICE)); net.eval()
    x = torch.tensor((OBS - F_MEAN) / F_STD, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        s = net.sample(x, n=n).cpu().numpy()[0]
    return s * TH_STD + TH_MEAN


# ---------- gold-standard posterior (emcee over the REBOUND likelihood) ----------
def log_prob(theta):
    m1, m2, h1, k1, h2, k2 = theta
    if not (S.M1_LO <= m1 <= S.M1_HI and S.M2_LO <= m2 <= S.M2_HI):
        return -np.inf
    if np.hypot(h1, k1) >= S.E1_MAX or np.hypot(h2, k2) >= S.E2_MAX:
        return -np.inf
    f = S.simulate(theta[None, :], noise_min=0.0)[0]         # deterministic forward model
    if np.isnan(f).any():
        return -np.inf
    return -0.5 * np.sum(((OBS - f) / NOISE_MIN) ** 2)        # Gaussian likelihood


def mcmc_posterior():
    p0 = THETA_TRUE + np.array([0.2, 0.5, 5e-3, 5e-3, 5e-3, 5e-3]) * np.random.default_rng(1).normal(
        size=(NWALKERS, S.THETA_DIM))
    # keep the initial ball inside the prior
    for i in range(NWALKERS):
        p0[i, 0] = np.clip(p0[i, 0], S.M1_LO + 1e-3, S.M1_HI - 1e-3)
        p0[i, 1] = np.clip(p0[i, 1], S.M2_LO + 1e-3, S.M2_HI - 1e-3)
    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(NWALKERS, S.THETA_DIM, log_prob, pool=pool)
        t0 = time.time()
        sampler.run_mcmc(p0, NSTEPS, progress=False)
        dt = time.time() - t0
    chain = sampler.get_chain(discard=NBURN, flat=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        neff = int((NSTEPS - NBURN) * NWALKERS / np.nanmean(tau))
    except Exception:
        neff = -1
    accept = float(np.mean(sampler.acceptance_fraction))
    return chain, dt, neff, accept


def make_figure(mdn, mcmc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for d, ax in enumerate(axes.ravel()):
        lo = min(mdn[:, d].min(), mcmc[:, d].min())
        hi = max(mdn[:, d].max(), mcmc[:, d].max())
        bins = np.linspace(lo, hi, 44)
        ax.hist(mcmc[:, d], bins=bins, density=True, color="#c0392b", alpha=0.45,
                label="MCMC (gold standard)")
        ax.hist(mdn[:, d], bins=bins, density=True, histtype="step", color="#1f6fb2",
                lw=2.0, label="MDN (amortized)")
        ax.axvline(THETA_TRUE[d], color="k", ls=":", lw=1.3)
        ax.set_title(S.THETA_NAMES[d], fontsize=11)
        ax.set_yticks([])
        if d == 0:
            ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Baseline parity: amortized MDN vs gold-standard MCMC on the same system "
                 "(timing-only, same REBOUND forward model)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("baseline_parity.png", dpi=145)
    print("  wrote baseline_parity.png")


def main():
    print("Gold-standard baseline parity (MCMC over our REBOUND likelihood vs the amortized MDN).")
    t0 = time.time(); mdn = mdn_posterior(); mdn_ms = (time.time() - t0) * 1000
    mcmc, mcmc_s, neff, accept = mcmc_posterior()
    print(f"  MDN inference {mdn_ms:.1f} ms  |  MCMC {mcmc_s:.0f} s "
          f"(accept {accept:.2f}, n_eff~{neff}), speed-up ~{mcmc_s*1000/mdn_ms:.0e}x\n")

    print(f"  {'param':>8} {'truth':>8} | {'MCMC mean':>10} {'MCMC std':>9} | "
          f"{'MDN mean':>10} {'MDN std':>9} | {'width MDN/MCMC':>14}")
    ratios = []
    for d in range(S.THETA_DIM):
        cm, cs = mcmc[:, d].mean(), mcmc[:, d].std()
        dm, ds = mdn[:, d].mean(), mdn[:, d].std()
        r = ds / (cs + 1e-12); ratios.append(r)
        print(f"  {S.THETA_NAMES[d]:>8} {THETA_TRUE[d]:>8.3f} | {cm:>10.3f} {cs:>9.3f} | "
              f"{dm:>10.3f} {ds:>9.3f} | {r:>14.2f}")
    print(f"\n  mean width ratio MDN/MCMC = {np.mean(ratios):.2f}  "
          f"(1.0 = amortized posterior matches the gold standard; <1 too tight, >1 too wide)")

    out = {"theta_true": THETA_TRUE.tolist(), "mcmc_seconds": round(mcmc_s, 1),
           "mdn_ms": round(mdn_ms, 1), "accept": round(accept, 3), "n_eff": neff,
           "per_param": {S.THETA_NAMES[d]: {
               "truth": float(THETA_TRUE[d]),
               "mcmc_mean": round(float(mcmc[:, d].mean()), 4), "mcmc_std": round(float(mcmc[:, d].std()), 4),
               "mdn_mean": round(float(mdn[:, d].mean()), 4), "mdn_std": round(float(mdn[:, d].std()), 4),
               "width_ratio": round(float(ratios[d]), 3)} for d in range(S.THETA_DIM)},
           "mean_width_ratio": round(float(np.mean(ratios)), 3)}
    json.dump(out, open("_baseline_parity.json", "w"), indent=2)
    print("  wrote _baseline_parity.json")
    make_figure(mdn, mcmc)


if __name__ == "__main__":
    main()
