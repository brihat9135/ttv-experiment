"""
TTV simulator: REBOUND (WHFast symplectic) N-body forward model — full element vector.

Maps a per-planet parameter vector theta -> transit times of BOTH planets -> a
fixed-length TTV feature. This is the "physics" we learn to invert.

  theta = (m1, m2, h1, k1, h2, k2)
     m1, m2 : planet masses [M_earth]
     h_i,k_i: eccentricity VECTOR of planet i, (e_i cos w_i, e_i sin w_i)
              Cartesian e-vectors avoid the angle wrap of w and the e->0 singularity,
              and are what TTVs actually constrain (Lithwick et al. 2012).

Geometry: 1 star + 2 coplanar planets (inclination 0, edge-on). Observer along +x; a
transit of a planet = its sky-plane coordinate y (relative to star) crosses 0 from -
to + while in front (x>0). BOTH planets transit, so each one's timing constrains the
other's mass. Periods and orbital phases are held fixed (periods are measured
directly from the transit ephemerides; the inner transit epoch sets t=0).

Observables: the timing feature is the O-C residual of each planet's transits. With
durations=True, simulate() additionally emits the transit-DURATION residual (TDV):
in this edge-on coplanar geometry the planet crosses the stellar diameter 2*R_STAR at
its sky-plane speed |v_y| at mid-transit, so the duration is 2*R_STAR/|v_y|. Duration
depends mainly on a planet's OWN orbital speed (set by its eccentricity), which
constrains eccentricity largely independently of the perturber mass -- the lever that
breaks the mass-eccentricity degeneracy the timing amplitude alone cannot.

With rv=True, simulate() additionally emits the star's RADIAL-VELOCITY curve: the
line-of-sight reflex velocity of the star (observer along +x, so RV = star's v_x in the
COM frame) sampled on a fixed epoch grid over the baseline, in m/s, mean-subtracted (the
systemic velocity is an unknown nuisance, fit out in practice). Unlike a duration -- a
single mid-transit speed that mostly pins the e-vector component h (=e cos w) -- the RV
curve's HARMONIC SHAPE (eccentric skew) and phase encode the FULL e-vector, so RV pins
k (=e sin w) too. That is the missing lever: durations constrain h, RV constrains k, and
near the 2:1 resonance the mass stays degenerate until BOTH are constrained.

Public API: simulate(theta, noise_min, rng, p2, durations, rv, rv_noise_ms),
sample_prior(n, rng), and the THETA_*/FEATURE_DIM/FEATURE_DIM_FULL/FEATURE_DIM_RV/
N_TRANSITS_*/N_RV constants. REBOUND runs one system per process, so simulate()
parallelizes across rows for batch generation.
"""
import os
import numpy as np
import rebound

G = 4.0 * np.pi**2            # AU, year, solar-mass units
MSUN = 1.0
MEARTH = 3.003e-6
DAY = 1.0 / 365.25
MIN = DAY / 1440.0

# ----- fixed, "known" system geometry -----
P1 = 10.0 * DAY               # inner planet period (measured)
P2 = 21.0 * DAY               # outer planet period (near 2:1, measured)
PHASE1 = 0.0                  # inner planet sky-position angle at t=0 (-> transits ~t=0)
PHASE2 = 1.0                  # outer planet sky-position angle at t=0 (rad, fixed)
BASELINE = 500.0 * DAY
STEP = P1 / 60.0              # WHFast timestep / transit-search sampling step
R_STAR = 0.00465047          # stellar radius [AU] (1 solar radius); sets transit duration
N_TRANSITS_1 = 40             # inner-planet transits used in the feature
N_TRANSITS_2 = 20            # outer-planet transits used in the feature
FEATURE_DIM = N_TRANSITS_1 + N_TRANSITS_2          # times-only feature (default)
DURATION_DIM = N_TRANSITS_1 + N_TRANSITS_2         # one duration residual per transit
FEATURE_DIM_FULL = FEATURE_DIM + DURATION_DIM      # times + durations (durations=True)
N_RV = 30                                          # radial-velocity samples over the baseline
RV_DIM = N_RV
RV_EPOCHS = np.linspace(0.0, BASELINE, N_RV)       # fixed RV cadence (years)
FEATURE_DIM_RV = FEATURE_DIM + DURATION_DIM + RV_DIM  # times + durations + RV (durations & rv)
# 1 AU/yr in m/s: line-of-sight reflex velocity unit conversion
AU_YR_TO_MS = 1.495978707e11 / (365.25 * 86400.0)  # ~4740.57 m/s per AU/yr

# ----- priors on the inferred parameters -----
M1_LO, M1_HI = 3.0, 15.0      # M_earth
M2_LO, M2_HI = 8.0, 45.0      # M_earth
E1_MAX = 0.12
E2_MAX = 0.12
THETA_DIM = 6
THETA_NAMES = ["m1 [Me]", "m2 [Me]", "h1", "k1", "h2", "k2"]

_MAXPROC = min(8, os.cpu_count() or 1)


def _add_planet(sim, m_earth, P, h, k, phase):
    """Add a planet from its eccentricity vector (h,k), placing it at sky-angle `phase`
    at t=0 (so its e-vector orientation is decoupled from its orbital phase)."""
    e = float(np.hypot(h, k))
    pomega = float(np.arctan2(k, h)) if e > 0 else 0.0
    f = phase - pomega                              # sky position angle pomega+f = phase
    sim.add(m=m_earth * MEARTH, P=P, e=e, pomega=pomega, f=f)


def _simulate_one(args):
    """Run one REBOUND system; return (times1, times2, durs1, durs2, rv) in years, or None
    if too few transits. durs_i are per-transit durations 2*R_STAR/|v_y| at mid-transit; rv
    is the star's line-of-sight reflex velocity (AU/yr) interpolated onto RV_EPOCHS, or None
    if rv not requested. args = (m1, m2, h1, k1, h2, k2, P2_years, collect_rv) — P2 is
    per-system (allows ratio sweeps)."""
    m1, m2, h1, k1, h2, k2, P2_i, collect_rv = args
    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=MSUN)
    _add_planet(sim, m1, P1, h1, k1, PHASE1)
    _add_planet(sim, m2, P2_i, h2, k2, PHASE2)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = STEP

    ps = sim.particles
    prev = [(ps[i].x - ps[0].x, ps[i].y - ps[0].y) for i in (1, 2)]
    t_prev = sim.t
    tmax = BASELINE * 1.8
    times = [[], []]
    durs = [[], []]
    rv_t, rv_v = [], []       # star's line-of-sight (v_x) reflex velocity samples over [0, BASELINE]
    need = (N_TRANSITS_1, N_TRANSITS_2)
    ESCAPE2 = 9.0      # (3 AU)^2: bound (even eccentric) orbits stay well below; ejections exceed it
    # collect transits until both planets have enough; if rv, also cover the baseline for RV sampling
    def more_needed():
        if len(times[0]) < need[0] or len(times[1]) < need[1]:
            return True
        return collect_rv and sim.t < BASELINE
    while more_needed() and sim.t < tmax:
        try:
            sim.integrate(sim.t + STEP)
        except Exception:
            return None                      # integrator blew up (collision / escape) -> unstable
        if collect_rv and sim.t <= BASELINE:
            rv_t.append(sim.t); rv_v.append(ps[0].vx)   # COM-frame star velocity = reflex RV
        for idx, pi in enumerate((1, 2)):
            x_new = ps[pi].x - ps[0].x
            y_new = ps[pi].y - ps[0].y
            if x_new*x_new + y_new*y_new > ESCAPE2:
                return None                  # planet ejected -> unstable system
            x_old, y_old = prev[idx]
            if y_old < 0.0 and y_new >= 0.0 and x_new > 0.0 and len(times[idx]) < need[idx]:
                frac = -y_old / (y_new - y_old)
                times[idx].append(t_prev + frac * STEP)
                # sky-plane crossing speed at mid-transit -> duration of the 2*R_STAR chord
                v_y = abs(ps[pi].vy - ps[0].vy) + 1e-12
                durs[idx].append(2.0 * R_STAR / v_y)
            prev[idx] = (x_new, y_new)
        t_prev = sim.t

    if len(times[0]) < need[0] or len(times[1]) < need[1]:
        return None
    rv = None
    if collect_rv:
        if len(rv_t) < 2 or rv_t[-1] < BASELINE * 0.99:
            return None                      # insufficient RV coverage of the baseline
        rv = np.interp(RV_EPOCHS, np.array(rv_t), np.array(rv_v))
    return (np.array(times[0][:need[0]]), np.array(times[1][:need[1]]),
            np.array(durs[0][:need[0]]),  np.array(durs[1][:need[1]]), rv)


def _oc(times, n):
    """transit times -> O-C residuals (minutes) about a fitted linear ephemeris."""
    times = np.asarray(times[:n])
    idx = np.arange(n)
    A = np.vstack([np.ones_like(idx), idx]).T.astype(float)
    (t0, P), *_ = np.linalg.lstsq(A, times, rcond=None)
    return (times - (t0 + idx * P)) / MIN


def _circ_duration(P):
    """transit duration of a circular orbit of period P [yr]: 2*R_STAR / v_circ.
    a^3 = P^2 in (AU, yr, Msun) units, v_circ = 2*pi*a/P, so T = R_STAR*P/(pi*a)."""
    a = P ** (2.0 / 3.0)
    return R_STAR * P / (np.pi * a)


def _dur_anom(durs, n, P):
    """transit durations -> anomaly (minutes) vs the circular-orbit duration for period P.
    Keeps BOTH the eccentricity-dependent mean offset (the strong signal) and the
    transit-to-transit variation (TDV). Referenced to a fixed physical baseline, not the
    per-system mean, so the mean offset is preserved."""
    durs = np.asarray(durs[:n])
    return (durs - _circ_duration(P)) / MIN


def _feature(pair, p2_i, durations=False, rv=False):
    if pair is None:
        return None
    t1, t2, d1, d2, rv_v = pair
    parts = [_oc(t1, N_TRANSITS_1), _oc(t2, N_TRANSITS_2)]
    if durations:
        parts.append(_dur_anom(d1, N_TRANSITS_1, P1))
        parts.append(_dur_anom(d2, N_TRANSITS_2, p2_i))
    if rv:
        rv_ms = rv_v * AU_YR_TO_MS                 # AU/yr -> m/s
        parts.append(rv_ms - rv_ms.mean())         # drop the unknown systemic velocity
    return np.concatenate(parts)


def _feature_dim(durations, rv):
    return FEATURE_DIM + (DURATION_DIM if durations else 0) + (RV_DIM if rv else 0)


def simulate(theta, noise_min=0.0, rng=None, p2=None, durations=False, rv=False,
             rv_noise_ms=1.0):
    """theta: (N,6). Returns features (timing/duration columns in minutes, RV columns in
    m/s); bad rows -> NaN. Columns are laid out as [timing | durations? | rv?]:
    durations=False, rv=False -> (N, FEATURE_DIM)       : O-C timing residuals only (default).
    durations=True,  rv=False -> (N, FEATURE_DIM_FULL)  : timing then duration (TDV) residuals.
    durations=True,  rv=True  -> (N, FEATURE_DIM_RV)    : timing, durations, then RV (m/s).
    p2: optional per-system outer period in years (N,); defaults to the global P2.
    noise_min (minutes) is added to the timing+duration columns; rv_noise_ms (m/s) to the
    RV columns."""
    theta = np.atleast_2d(theta).astype(float)
    n = theta.shape[0]
    if p2 is None:
        p2 = np.full(n, P2)
    else:
        p2 = np.asarray(p2, dtype=float)
    args = [(*theta[i], p2[i], rv) for i in range(n)]

    if n >= 64 and _MAXPROC > 1:
        import multiprocessing as mp
        with mp.Pool(_MAXPROC) as pool:
            out = pool.map(_simulate_one, args, chunksize=max(1, n // (_MAXPROC * 4)))
    else:
        out = [_simulate_one(a) for a in args]

    dim = _feature_dim(durations, rv)
    feats = np.full((n, dim), np.nan)
    for k, pair in enumerate(out):
        f = _feature(pair, p2[k], durations=durations, rv=rv)
        if f is not None:
            feats[k] = f
    if noise_min > 0 or (rv and rv_noise_ms > 0):
        rng = rng or np.random.default_rng()
        n_time_dur = FEATURE_DIM + (DURATION_DIM if durations else 0)
        if noise_min > 0:
            feats[:, :n_time_dur] += rng.normal(0.0, noise_min, size=(n, n_time_dur))
        if rv and rv_noise_ms > 0:
            feats[:, n_time_dur:] += rng.normal(0.0, rv_noise_ms, size=(n, RV_DIM))
    return feats


def sample_prior(n, rng=None):
    rng = rng or np.random.default_rng()
    m1 = rng.uniform(M1_LO, M1_HI, n)
    m2 = rng.uniform(M2_LO, M2_HI, n)
    # uniform-in-disk eccentricity vectors (area-uniform, not 1/e-biased)
    e1 = E1_MAX * np.sqrt(rng.uniform(0, 1, n)); w1 = rng.uniform(0, 2*np.pi, n)
    e2 = E2_MAX * np.sqrt(rng.uniform(0, 1, n)); w2 = rng.uniform(0, 2*np.pi, n)
    return np.stack([m1, m2, e1*np.cos(w1), e1*np.sin(w1), e2*np.cos(w2), e2*np.sin(w2)], axis=1)


if __name__ == "__main__":
    th = sample_prior(6, np.random.default_rng(0))
    f = simulate(th, noise_min=0.5, rng=np.random.default_rng(1))
    fd = simulate(th, noise_min=0.5, rng=np.random.default_rng(1), durations=True)
    frv = simulate(th, noise_min=0.5, rng=np.random.default_rng(1), durations=True, rv=True)
    print("rebound", rebound.__version__, "| times dim:", f.shape[1],
          "| times+dur dim:", fd.shape[1], "| times+dur+rv dim:", frv.shape[1])
    print("inner TTV  rms (min):", np.round(np.nanstd(f[:, :N_TRANSITS_1], 1), 2))
    print("outer TTV  rms (min):", np.round(np.nanstd(f[:, N_TRANSITS_1:], 1), 2))
    d1 = fd[:, FEATURE_DIM:FEATURE_DIM + N_TRANSITS_1]
    d2 = fd[:, FEATURE_DIM + N_TRANSITS_1:]
    print("inner TDV  rms (min):", np.round(np.nanstd(d1, 1), 2))
    print("outer TDV  rms (min):", np.round(np.nanstd(d2, 1), 2))
    rvcols = frv[:, FEATURE_DIM_FULL:]
    print("star RV semi-amp (m/s):", np.round((np.nanmax(rvcols, 1) - np.nanmin(rvcols, 1)) / 2, 2))
