"""
Multi-seed the in-hybrid RV result, stage 1 (jaxttv venv): generate 3 INDEPENDENT cross-resonance
timing+RV training sets (distinct random draws) so we can put an error bar on the ~83% separatrix
tightening. The shared held-out test set (jaxttv_resrv_test.npz, seed 2) is reused across seeds.

    /tmp/jaxttv-venv/bin/python gen_jaxttv_resrv_multiseed.py  ->  jaxttv_resrv_{train,val}_s{0,1,2}.npz
"""
import numpy as np
from gen_jaxttv_resonance_rv import generate

SEEDS = [0, 1, 2]
for s in SEEDS:
    th, rt, oc, rv = generate(600, 300 + s)
    np.savez(f"jaxttv_resrv_train_s{s}.npz", theta=th, ratio=rt, oc=oc, rv=rv)
    th, rt, oc, rv = generate(120, 400 + s)
    np.savez(f"jaxttv_resrv_val_s{s}.npz", theta=th, ratio=rt, oc=oc, rv=rv)
    print(f"seed {s}: train/val saved")
print("done. shared test = jaxttv_resrv_test.npz")
