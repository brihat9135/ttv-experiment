"""
Two-planet TTV demo (pure numpy N-body, vectorized over many systems at once).

Physics: 1 star + 2 coplanar planets, full gravitational N-body, leapfrog integrator.
Planet 1 is the "clock" — we record every time it transits (crosses the +x axis,
where the observer sits). Planet 2 (the perturber) tugs on it, so the transits are
NOT perfectly periodic. Those deviations are the TTVs.

We do two things:
  (A) Simulate ONE "true" system and show what the data looks like:
      a list of transit times -> the wiggle (O-C) that encodes planet 2.
  (B) Sweep a grid of (perturber mass m2, perturber eccentricity e2), simulate ALL of
      them at once, and score each against the true transit times. The set of
      combinations that fit well = the "cloud of answers" = the degeneracy.
"""
import numpy as np

# ---- units: AU, years, solar masses; G = 4 pi^2 ----
G = 4.0 * np.pi**2
MSUN = 1.0
MEARTH = 3.003e-6              # in solar masses
DAY = 1.0 / 365.25            # one day in years
MIN = DAY / 1440.0           # one minute in years

def elements_to_state(M_star, a, e, omega, nu):
    """Orbital elements -> (position, velocity) in the orbital plane. Vectorized."""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    # perifocal frame (periapsis along local x), then rotate by omega
    cos_wn = np.cos(omega + nu); sin_wn = np.sin(omega + nu)
    pos = np.stack([r * cos_wn, r * sin_wn], axis=-1)
    mu = G * M_star
    vfac = np.sqrt(mu / p)
    # v in perifocal: (-sin nu, e + cos nu), then rotate by omega
    vx_pf = -np.sin(nu); vy_pf = e + np.cos(nu)
    cw = np.cos(omega); sw = np.sin(omega)
    vx = vfac * (cw * vx_pf - sw * vy_pf)
    vy = vfac * (sw * vx_pf + cw * vy_pf)
    vel = np.stack([vx, vy], axis=-1)
    return pos, vel

def setup(G_sys, m1, m2, P1, P2, e2, nu2_0):
    """Build G_sys independent systems. Arrays broadcast over the grid dimension.
    Returns r,v of shape (G_sys,3,2) and masses (G_sys,3). Bodies: 0=star,1=p1,2=p2."""
    m1 = np.broadcast_to(m1, (G_sys,)).astype(float)
    m2 = np.broadcast_to(m2, (G_sys,)).astype(float)
    e2 = np.broadcast_to(e2, (G_sys,)).astype(float)
    P1 = np.broadcast_to(P1, (G_sys,)).astype(float)
    P2 = np.broadcast_to(P2, (G_sys,)).astype(float)
    nu2_0 = np.broadcast_to(nu2_0, (G_sys,)).astype(float)

    a1 = (G * MSUN * P1**2 / (4*np.pi**2))**(1/3)
    a2 = (G * MSUN * P2**2 / (4*np.pi**2))**(1/3)

    # planet 1: e=0, omega=0, starts ON the +x axis (nu=0) -> transits at t~0
    r1, v1 = elements_to_state(MSUN, a1, np.zeros(G_sys), np.zeros(G_sys), np.zeros(G_sys))
    # planet 2: eccentric, omega=0, phase nu2_0
    r2, v2 = elements_to_state(MSUN, a2, e2, np.zeros(G_sys), nu2_0)

    # star placed so center-of-mass is at rest at the origin
    r0 = -(m1[:, None]*r1 + m2[:, None]*r2) / MSUN
    v0 = -(m1[:, None]*v1 + m2[:, None]*v2) / MSUN

    r = np.stack([r0, r1, r2], axis=1)          # (G_sys,3,2)
    v = np.stack([v0, v1, v2], axis=1)
    m = np.stack([np.full(G_sys, MSUN), m1, m2], axis=1)  # (G_sys,3)
    return r, v, m

def accel(r, m):
    """Pairwise gravitational acceleration. r:(G,3,2) m:(G,3) -> a:(G,3,2)."""
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[:, j, :] - r[:, i, :]               # (G,2)
            dist3 = (np.sum(d*d, axis=1) + 1e-18)**1.5  # (G,)
            a[:, i, :] += G * m[:, j, None] * d / dist3[:, None]
    return a

def integrate(r, v, m, t_end, dt):
    """Leapfrog. Records planet-1 transits (relative-to-star +x axis crossing, y:-->+)."""
    G_sys = r.shape[0]
    a = accel(r, m)
    t = 0.0
    # transit times per system, collected as lists
    transits = [[] for _ in range(G_sys)]
    # previous relative position of planet 1
    rel_prev = r[:, 1, :] - r[:, 0, :]
    nsteps = int(np.ceil(t_end / dt))
    for _ in range(nsteps):
        v = v + 0.5*dt*a
        r = r + dt*v
        a = accel(r, m)
        v = v + 0.5*dt*a
        t += dt
        rel = r[:, 1, :] - r[:, 0, :]
        # crossing +x axis: y goes - -> + with x>0
        yprev = rel_prev[:, 1]; ynow = rel[:, 1]; xnow = rel[:, 0]
        cross = (yprev < 0) & (ynow >= 0) & (xnow > 0)
        if cross.any():
            frac = -yprev / (ynow - yprev)            # linear interp to y=0
            t_tr = (t - dt) + frac*dt
            for k in np.nonzero(cross)[0]:
                transits[k].append(t_tr[k])
        rel_prev = rel
    return transits

def linear_ephemeris(times):
    """Fit t_n = t0 + n*P; return (t0, P, residuals_in_minutes)."""
    times = np.asarray(times)
    n = np.arange(len(times))
    A = np.vstack([np.ones_like(n), n]).T.astype(float)
    (t0, P), *_ = np.linalg.lstsq(A, times, rcond=None)
    resid = times - (t0 + n*P)
    return t0, P, resid / MIN   # minutes

# ============================== (A) THE "TRUE" SYSTEM ==============================
m1_true = 5*MEARTH
m2_true = 25*MEARTH
P1_true = 10.0*DAY
P2_true = 21.0*DAY            # near (but not exactly) the 2:1 resonance with planet 1
e2_true = 0.03
nu2_true = 1.0               # initial phase of perturber (radians)
BASELINE = 600.0*DAY
DT = P1_true/300.0

r, v, m = setup(1, m1_true, m2_true, P1_true, P2_true, e2_true, nu2_true)
truth = integrate(r, v, m, BASELINE, DT)[0]
t0, P, ttv = linear_ephemeris(truth)

print("="*70)
print("(A) THE 'TRUE' SYSTEM  —  this is what a telescope would hand us")
print("="*70)
print(f"Planet 1 (the clock): mass = {m1_true/MEARTH:.0f} M_earth, period ~ {P/DAY:.4f} days")
print(f"Planet 2 (perturber): mass = {m2_true/MEARTH:.0f} M_earth, P2 ~ {P2_true/DAY:.1f} d, e2 = {e2_true}")
print(f"\nNumber of recorded transits of planet 1: {len(truth)}")
Pttv = 1.0/abs(2/P2_true - 1/P1_true)
print(f"Predicted TTV 'super-period' = 1/|2/P2 - 1/P1| = {Pttv/DAY:.0f} days")

print("\nFirst 8 transit times (days) and their TTV deviation (minutes):")
print(f"{'n':>3} {'transit time (d)':>18} {'O-C (min)':>12}")
for n in range(min(8, len(truth))):
    print(f"{n:>3} {truth[n]/DAY:>18.5f} {ttv[n]:>12.2f}")

# crude ASCII O-C plot
print("\nO-C diagram (the TTV signal — vertical = early/late in minutes):")
amp = max(1e-9, np.max(np.abs(ttv)))
width = 51; mid = width//2
for n in range(len(truth)):
    col = int(round(mid + ttv[n]/amp*(mid-1)))
    line = [' ']*width; line[mid] = '|'; line[col] = '*'
    print(f"  t={truth[n]/DAY:7.1f}d  {''.join(line)}")
print(f"  (horizontal scale: +/- {amp:.1f} minutes full width)")

# ============================== (B) THE DEGENERACY ==============================
print("\n" + "="*70)
print("(B) INFERENCE: which (mass, eccentricity) pairs fit the SAME data?")
print("="*70)
nM, nE = 13, 13
m2_grid = np.linspace(10, 40, nM)*MEARTH        # perturber mass, M_earth
e2_grid = np.linspace(0.0, 0.10, nE)            # perturber eccentricity
MM, EE = np.meshgrid(m2_grid, e2_grid, indexing='ij')
flatM = MM.ravel(); flatE = EE.ravel()
Gs = flatM.size

r, v, m = setup(Gs, m1_true, flatM, P1_true, P2_true, flatE, nu2_true)
grid_tr = integrate(r, v, m, BASELINE, DT)

# chi-square vs truth (assume same transit count; align by index)
sigma = 0.5*MIN     # assumed timing precision: 0.5 minute
ntr = min(len(truth), min(len(g) for g in grid_tr))
truth_arr = np.array(truth[:ntr])
chi2 = np.full(Gs, np.nan)
for k in range(Gs):
    g = np.array(grid_tr[k][:ntr])
    chi2[k] = np.sum(((g - truth_arr)/sigma)**2) / ntr   # reduced-ish
chi2 = chi2.reshape(nM, nE)

print(f"\nFit quality across the grid (lower = better fit to the true data).")
print(f"Each cell = one simulated 2-planet system. '.'=great fit, '#'=terrible.")
print(f"Truth is m2={m2_true/MEARTH:.0f} Mearth, e2={e2_true}.\n")
ramp = " .:-=+*#@"
lo = np.nanmin(chi2)
print("        e2:  " + " ".join(f"{e:4.2f}" for e in e2_grid))
for i in range(nM):
    cells = []
    for j in range(nE):
        v_ = chi2[i, j] - lo
        idx = min(len(ramp)-1, int(v_/ (np.nanmax(chi2)-lo+1e-9) * (len(ramp)-1)))
        cells.append(ramp[idx])
    print(f"m2={m2_grid[i]/MEARTH:5.1f}  " + "    ".join(cells))
print("\nThe band of light cells = combinations that all fit the data almost")
print("equally well. THAT band is the degeneracy a good posterior must capture.")
