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
| `simulator.py` | REBOUND (WHFast) N-body forward model; per-system P2; ejection detection; TTV feature extraction |
| `model.py` | Mixture Density Network = amortized neural posterior estimator |
| `train.py`, `evaluate.py` | train the 6-D posterior; recovery + calibration (SBC/coverage) + speed |
| `ttv_demo.py` | standalone conceptual demo (TTV signal + degeneracy valley) |
| `resonance_sweep.py`, `_ecc.py` | drive toward 2:1 at fixed ratio (e≤0.12, e≤0.30) |
| `cross_resonance.py` | ONE model spanning ratio 1.9–2.2 (ratio as conditioning input) |
| `cross_resonance_chaos.py` | same, giant+eccentric (strong-interaction) regime |
| `disambiguation.py` | ABC reference vs MDN width → physical-vs-model test |
| `tf_pretrain.py` | transformer encoder + masked pretraining → sim-to-real gap test |

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

## 4. Refined thesis (what the evidence supports)

- Amortized SBI for TTV is **calibration-robust out of the box** — even at 160-min TTVs,
  giant eccentric planets, and across the resonance. It does **not** fail by overconfidence.
- It fails by **unstable training** and **informativeness collapse** when spanning the
  resonance — and that collapse is **largely physical (data-limited)**, not a model defect.
- Therefore the bottleneck for posterior *sharpness* is **observational, not algorithmic**.
- The genuine ML contributions are: **(a) amortization at survey (PLATO) scale**;
  **(b) a normalizing flow** for faithful *multimodal* posteriors + *stable training*
  (not for tighter marginals); **(c) ingesting more observables** (transit durations TDV,
  RV) — the real lever for information; **(d) sim-to-real robustness** via pretraining.

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

1. **[½ day] Middle path:** extend `simulator.py` to emit transit *durations* (and depths);
   add to the feature vector; rerun `disambiguation.py`-style check → does sharpness improve?
2. **[1 day] Flow head:** swap MDN → normalizing flow (`sbi` toolkit / neural spline flow);
   re-test cross-resonance training stability + multimodal faithfulness.
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
