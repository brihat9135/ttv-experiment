# TTV-experiment — Handoff / "pick it up here"

A self-contained map of the whole investigation: the goal, what was built, every
experiment and its verdict, the refined thesis, the literature landscape, and the open
forks with concrete next steps. Detailed results live in `README.md`; this is the resume
point.

---

## 1. The goal we started from

Use machine learning to **speed up posterior inference of exoplanet masses/orbits from
transit-timing variations (TTVs)** in dynamically-interacting systems — replacing
day-long per-system MCMC/N-body fits with fast, *calibrated* amortized inference, and
(the original hope) handling the resonant/chaotic regime that prior methods exclude.

## 2. What was built (all runnable, numpy + torch + rebound)

| File | Role |
|---|---|
| `simulator.py` | REBOUND (WHFast) N-body forward model; per-system P2; ejection detection; TTV feature extraction. **Emits transit DURATIONS (`durations=True` → 120-D) and the star's RADIAL-VELOCITY curve (`rv=True` → +30-D, 150-D total): line-of-sight reflex velocity (m/s) on a fixed epoch grid, mean-subtracted. Backward compatible (default 60-D timing-only).** |
| `model.py` | Mixture Density Network = amortized neural posterior estimator |
| `train.py`, `evaluate.py` | train the 6-D posterior; recovery + calibration (SBC/coverage) + speed |
| `ttv_demo.py` | standalone conceptual demo (TTV signal + degeneracy valley) |
| `resonance_sweep.py`, `_ecc.py` | drive toward 2:1 at fixed ratio (e≤0.12, e≤0.30) |
| `cross_resonance.py` | ONE model spanning ratio 1.9–2.2 (ratio as conditioning input) |
| `cross_resonance_chaos.py` | same, giant+eccentric (strong-interaction) regime |
| `disambiguation.py` | ABC reference vs MDN width → physical-vs-model test |
| `tf_pretrain.py` | transformer encoder + masked pretraining → sim-to-real gap test |
| `observables_experiment.py` | **A/B: timing-only vs timing+durations (fixed ratio). The observables test.** |
| `posterior_overlay.py` | **before/after mass–ecc posterior figure (degeneracy ridge → tight blob).** |
| `robustness_durations.py` | **durations result vs coarser duration noise (0.5–10 min), 3 seeds.** |
| `observables_cross_resonance.py` | **timing vs timing+durations spanning the resonance, 3 seeds.** |
| `observables_rv.py` | **3-arm A/B/C spanning the resonance: timing / +durations / +durations+RV, 3 seeds. The k-constraint test.** |
| `robustness_rv.py` | **RV result vs coarser RV precision (0.3–10 m/s), spanning the resonance, 3 seeds.** |
| `cadence_rv.py` | **RV result vs number of RV epochs (3–30, subsampled), spanning the resonance, 3 seeds.** |

Two public blog posts write this up: *Calibrated TTV Inference* (the diagnosis) and *Breaking the
TTV Degeneracy* (the durations result), at github.com/brihat9135/brihat-ai.

Method throughout: **amortized Neural Posterior Estimation (NPE)** — train a network on
simulated (θ, data) pairs to output p(θ|data). θ = (m1, m2, h1, k1, h2, k2): two masses
+ two eccentricity vectors. Periods/phases treated as measured.

## 3. Experiments and verdicts (chronological)

1. **PoC (2 params, m2+e2):** calibrated posterior, recovers the mass–eccentricity
   degeneracy, ~10⁴× faster than a brute-force grid. ✅
2. **REBOUND upgrade:** matched the toy integrator (super-period, amplitude, mass-scaling). ✅
3. **Full 6-param element vector, both planets transit:** calibrated across **all 6**
   params (coverage ≈ nominal, flat SBC). ✅
4. **Resonance sweeps at fixed ratio** (e≤0.12 and e≤0.30): **no calibration break** —
   graceful degradation; posteriors widen honestly. Instability stayed ~0% and was
   *lower* near resonance → **2:1 resonance is dynamically protective**. Hypothesis
   "calibration breaks at the separatrix" → **falsified**.
5. **Cross-resonance (one model spanning 1.9–2.2):** the real strain appears —
   **unstable training** (val NLL diverges; only early-stopping saves it) and
   **informativeness collapse** (best val NLL −5.9 fixed-ratio → +3.0 → +5.7 with
   giants). Separatrix calibration drift is modest/noisy (~6%, under the 7% flag).
6. **Disambiguation (ABC reference vs MDN):** the informativeness collapse is **largely
   PHYSICAL (data-limited)** — the MDN's widths track an independent likelihood-based
   reference and are never wider; combined with calibration evidence, the MDN is
   *faithful*, not over-cautious. → A better estimator won't sharpen marginals; the
   lever is **more data/observables**. (Caveat: ABC was sparsity-limited; leans on the
   calibration argument.)
7. **Sim-to-real masked pretraining (`tf_pretrain.py`):** controlled gap (clean vs
   correlated/heavy-tailed noise). Masked pretraining **helped calibration robustness**
   (calErr 5.4%→2.4%, frozen encoder) but **not point accuracy** (RMSE slightly worse).
   So pretraining = a **robustness** tool, not an information tool. (Single seed —
   suggestive, not proven.)
8. **Observables: add transit DURATIONS (`observables_experiment.py` etc.):** the lever
   from §6, now demonstrated. Duration ≈ 2·R★/|v_y| at mid-transit measures the
   eccentricity-vector component **h** (=e·cosϖ) almost independently of perturber mass
   (corr(h, duration) ≈ −1). **Fixed ratio, e≤0.12:** A/B (timing vs timing+dur) on the
   same systems/noise → mass posteriors tighten **76–81%**, h-components **94–95%**, best
   val NLL −4.3 → −10.2, coverage stays ~nominal. ✅ **Robustness (3 seeds, duration noise
   0.5→10 min):** survives — at 10-min precision m2 still ~half the timing-only width;
   degrades gracefully because the signal is in the *mean* anomaly (√N averaging). ✅
   **Across the resonance (3 seeds, ratio 1.9–2.2, e≤0.15):** durations **partially**
   rescue the collapse: val NLL +3.6 → −1.4 (~5 nats back), h-components pinned (93–96%),
   calibration holds, **but mass tightens only ~22%** (weakest at the separatrix) because
   durations constrain **h, not k**; with k unconstrained the near-resonant mass stays
   partly degenerate. → **More observables is the lever, but "more observables" ≠ "any one
   observable": the next one must constrain k (RV, or longer baseline / more transits).**
9. **Constrain k with RADIAL VELOCITY (`observables_rv.py`):** the direct sequel to §3.8,
   acting on its own diagnosis. The star's line-of-sight reflex-velocity curve has amplitude
   proportional to planet mass (a near-direct mass measurement, *independent* of the TTV
   degeneracy) and an eccentric **harmonic shape + phase** that encode the **full** e-vector,
   so RV pins **k** as well as mass. Linear probe (R² on the outer planet) confirms the lever:
   k2 from durations 0.16 to RV **0.63**; m2 from RV alone **0.99**. **3-arm A/B/C spanning the
   resonance (ratio 1.9–2.2, e≤0.15, 3 seeds):** A timing / B +durations / C +durations+RV.
   Best val NLL **+3.6 → −1.4 → −5.8** (RV adds ~4.4 nats on top of durations' ~5). The
   headline: **near-resonance (separatrix) m2 mass tightening jumps from durations' 16% to RV's
   83%** — back to the ~80% seen at fixed ratio in §3.8. **k is finally pinned** — durations did
   ~nothing for k (k1 +5%, k2 −16%), RV pins it (k1 57%, k2 51%) — while h stays pinned
   (96–97%). Coverage holds roughly nominal but drifts mildly overconfident in the richer arm
   (90% interval: timing 0.88 to +dur+RV 0.83). → **The §3.8 prediction is confirmed:
   durations constrain h, RV constrains k, and only with BOTH does the near-resonant mass
   degeneracy break.** ✅ **Robustness (`robustness_rv.py`, 3 seeds, RV noise 0.3→10 m/s vs the
   timing+durations reference):** the near-resonance rescue degrades **gracefully and
   monotonically** — separatrix m2 tightening 91% @0.3 m/s, 81% @1, 56% @3, washing out to 8%
   @10 m/s (where noise meets the ~6–12 m/s signal amplitude); k2 width tracks it (0.26→0.88);
   coverage stays flat ~0.85–0.88 (no overconfidence as precision coarsens). So the result
   holds at realistic ESPRESSO/HARPS precision (≲1–3 m/s). ✅ **Cadence (`cadence_rv.py`, 3
   seeds, RV epochs subsampled 3→30 at 1 m/s):** a clean **mass-vs-k split** — mass lives in the
   RV *amplitude*, so just **3 epochs already buy 50%** near-resonance m2 tightening (rising
   64/74/81% at 8/15/30); but **k lives in the eccentric harmonic shape**, so k2 lags (23% @3 →
   55% @30 epochs) and needs the dense campaign. Coverage flat ~0.85–0.88 at every cadence. →
   **A handful of RV visits cheaply recovers most of the near-resonant MASS; fully pinning the
   eccentricity-vector orientation (k) is what costs epochs.** ✅

## 4. Refined thesis (what the evidence supports)

- Amortized SBI for TTV is **calibration-robust out of the box** — even at 160-min TTVs,
  giant eccentric planets, and across the resonance. It does **not** fail by overconfidence.
- It fails by **unstable training** and **informativeness collapse** when spanning the
  resonance — and that collapse is **largely physical (data-limited)**, not a model defect.
- Therefore the bottleneck for posterior *sharpness* is **observational, not algorithmic**.
  **Now directly demonstrated end-to-end (§3.8–3.9):** transit durations break the degeneracy
  (76–81% mass tightening at fixed ratio) by pinning **h**; near resonance that is only a
  partial fix (~16–24%) because durations leave **k** free; **adding radial velocity pins k
  (and mass directly), and near-resonance mass tightening snaps from 16% to 83%** — back to
  the fixed-ratio level. The observational diagnosis is confirmed twice over: it is not just
  "more data" but **the right data — h from durations AND k from RV.**
- The genuine ML contributions are: **(a) amortization at survey (PLATO) scale**;
  **(b) a normalizing flow** for faithful *multimodal* posteriors + *stable training*
  (not for tighter marginals); **(c) ingesting more observables** (transit durations TDV
  ✅ done, h; radial velocity ✅ done, k + mass), the real lever for information, now shown
  to break the near-resonant mass degeneracy; **(d) sim-to-real robustness** via pretraining.

The original "chaos-aware emulation" framing is replaced by: **stable, faithful,
amortized SBI that jointly exploits all observables, validated by calibration at scale.**

## 5. Literature landscape (two scans: original + foundation-model)

**TTV / mass inference:** TTVFast/TTVFaster (fast forward models, break at resonance);
Nbody-AI / DeepTTV / LSTM (point/prior estimators, exclude resonance, uncalibrated);
differentiable N-body + HMC (Agol "Photodynamics II" 2410.03874; `jaxttv`/NumPyro;
Ofir 2025) — **this is the incumbent fast-posterior method and our baseline to beat**.
SBI/NPE+normalizing-flows are mature in **atmospheric retrieval, gravitational waves,
pulsar timing** — but **not applied to TTV**.

**Light-curve foundation / transformer models (2025–26):** **FALCO** (2504.20290, AJ 2026)
— self-supervised transformer on Kepler light curves; downstream = *stellar* tasks
(variability, log g, flares); **no planets/mass/TTV**; code availability **unconfirmed**.
**ExoVeil** (2606.02778, 2026) — transformer "world model" on ~16.5k Kepler LCs for
single-transit **detection**; **open-source with pretrained weights**; no mass/dynamics.
Plus several transformer transit **detectors/classifiers** (TESS FFI, false-positive ID).

**The open niche (triangulated, still appears unclaimed):**
> **Amortized, calibration-validated posterior inference of planet masses/orbits** —
> either from TTVs (Path A) or end-to-end from light curves via a foundation backbone
> (Path B). The two ecosystems (LC foundation models ↔ TTV mass inference) are disjoint;
> nobody connects "foundation encoder → calibrated dynamical-mass posterior."

Caveat: absence of evidence in focused scans, not proof. SBI is hot; verify with a fresh
arXiv `astro-ph.EP` check before staking novelty.

## 6. The forks (decide based on appetite for scope)

- **Path A — pragmatic TTV-SBI (small, fundable):** keep transit-times→masses, but add
  **more observables** (TDV, depths) to the forward model + a **normalizing-flow** head;
  validate calibration at scale and **benchmark against `jaxttv` HMC** on Kepler-9 /
  TRAPPIST-1 / TOI-216. Directly acts on the data-limit finding.
- **Path B — foundation-backbone → mass posterior (big, novel):** fine-tune a light-curve
  foundation encoder (FALCO-style, or build on **ExoVeil's open weights**) with our
  **calibrated SBI head** to infer masses end-to-end from photometry. Uses strictly more
  information than transit times. Expectation (from §3.7): the backbone mainly buys
  **robustness**, not sharper masses. Bigger compute/data engineering.
- **Middle path (cheapest high-value):** add **TDV + transit depths** to the current
  forward model and show the near-resonance posterior tightens — confirms the physical
  verdict *and* demonstrates the real information lever, without foundation-model machinery.

## 7. Concrete "pick it up here" next steps

0. ✅ **DONE, Middle path (durations):** `simulator.py` emits durations; A/B + robustness
   + cross-resonance run (§3.8). Verdict: breaks the degeneracy at fixed ratio, partial fix
   near resonance (h yes, k no). Two posts published. **This is the current frontier.**
1. ✅ **DONE, Constrain k (radial velocity):** `simulator.py` emits the star's RV curve
   (`rv=True`); `observables_rv.py` runs the 3-arm A/B/C across the resonance (§3.9). Verdict:
   RV pins **k** (and mass directly), and near-resonance mass tightening jumps from durations'
   **16% to 83%** — confirming the §3.8 prediction. ✅ **Robustness done** (`robustness_rv.py`):
   the rescue degrades gracefully with RV precision (91/81/56/8% tightening @ 0.3/1/3/10 m/s),
   holding at realistic ≲1–3 m/s. ✅ **Cadence done** (`cadence_rv.py`): mass-vs-k split — 3 RV
   epochs already buy 50% near-resonance mass tightening, but k needs the dense ~30-epoch
   campaign (k2 23%→55%). **This is the current frontier.**
   *Next sub-steps from here:* (a) a posterior-overlay figure for the near-resonant m2–k2 plane
   (B vs C) like `posterior_overlay.py`; (b) write it up as a third post / extend the durations
   post; (c) the bigger forks in §6 (flow head, jaxttv baseline parity).
2. **[1 day] Flow head:** swap MDN → normalizing flow (`sbi` toolkit / neural spline flow);
   re-test cross-resonance training stability + multimodal faithfulness. (Targets the
   training-instability defect that durations did NOT fix.)
3. **[1 day] Baseline parity:** install `jaxttv`; get an HMC posterior on one real-ish
   near-resonance system; compare width/shape to our amortized posterior (gold-standard check).
4. **[½ day] Firm up §3.7:** rerun sim-to-real arms across 3–5 seeds → is the 5.4%→2.4%
   calibration recovery real?
5. **[½ day] Novelty lock:** fresh arXiv `astro-ph.EP` scan for SBI-for-TTV and
   foundation→mass; confirm FALCO/ExoVeil code/weights availability.
6. **[stretch] Path B prototype:** ExoVeil/FALCO encoder embeddings → MDN/flow head →
   masses on simulated transits; measure whether LC embeddings beat hand-made O-C features.

## 8. Honest open questions / risks

- **Incumbent:** differentiable N-body + HMC (`jaxttv`) is mature and exact per-system;
  our edge must be **amortization at scale** + **multimodal/calibrated** posteriors, not
  raw per-system accuracy. Benchmark early.
- **Information limit:** sharpness is data-limited near resonance (strongly supported, not
  proven — ABC was sparsity-limited). Don't expect a model to fix it.
- **Foundation transfer (Path B):** unknown whether FALCO/ExoVeil representations (tuned
  for stellar variability / detection) encode the transit-level detail dynamics needs.
- **Reproducibility:** key results are single-seed; multi-seed before any write-up.
- **Sim-to-real gap was small** in our synthetic test — the robustness motivation is real
  but modest; validate on actual Kepler systematics before over-investing.
