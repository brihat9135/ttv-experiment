"""
Fetch REAL Kepler-9 b/c transit-timing data from the Holczer et al. 2016 Kepler TTV catalog
(VizieR J/ApJS/225/9, table3: per-transit O-C in minutes). Kepler-9b = KOI 377.01 (inner,
P~19.2 d), Kepler-9c = KOI 377.02 (outer, P~39.0 d). Dynamical masses from the literature:
43.4 / 29.8 M_earth (Borsato+2019). This is the real-data input for the phase-6 validation.

Parses to a tidy CSV (planet, N, tn_BJD, OC_min, eOC_min), keeping only clean transits
(no flag, not an outlier, not overlapping). Reproducible: re-downloads from VizieR if the raw
TSVs are absent. Writes kepler9_ttv.csv and prints a summary + derived linear periods.
"""
import os, urllib.request, numpy as np

KOIS = {"377.01": "Kepler-9b", "377.02": "Kepler-9c"}
BASE = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/225/9/table3"
        "&KOI={koi}&-out=KOI,N,tn,O-C,e_O-C,f_O-C,Out,Over&-out.max=99999")


def fetch(koi):
    path = f"kepler9_koi{koi}.tsv"
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE.format(koi=koi), path)
    return path


def parse(path):
    N, tn, oc, eoc = [], [], [], []
    for line in open(path):
        if not (line[:1].isdigit() or line[:1] == " ") or "\t" not in line:
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 8 or not p[1].strip() or not p[3].strip():
            continue
        try:
            n, t, o, e = int(p[1]), float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        flag, out, over = p[5].strip(), p[6].strip(), p[7].strip()
        if flag == "" and out in ("", "0") and over in ("", "0"):
            N.append(n); tn.append(t); oc.append(o); eoc.append(e)
    return np.array(N), np.array(tn), np.array(oc), np.array(eoc)


def main():
    rows = ["planet,N,tn_BJD,OC_min,eOC_min"]
    print("Real Kepler-9 TTVs (Holczer+2016, VizieR J/ApJS/225/9 table3):")
    periods = {}
    for koi, name in KOIS.items():
        N, tn, oc, eoc = parse(fetch(koi))
        # linear period from expected mid-times tn vs transit number N
        A = np.vstack([np.ones_like(N), N]).T.astype(float)
        (t0, P), *_ = np.linalg.lstsq(A, tn, rcond=None)
        periods[name] = P
        for n, t, o, e in zip(N, tn, oc, eoc):
            rows.append(f"{name},{n},{t:.6f},{o:.4f},{e:.4f}")
        print(f"  {name} (KOI {koi}): {len(N)} clean transits | P = {P:.4f} d | "
              f"O-C span {(oc.max()-oc.min())/1440:.2f} d | med err {np.median(eoc):.2f} min")
    open("kepler9_ttv.csv", "w").write("\n".join(rows) + "\n")
    print(f"  period ratio Pc/Pb = {periods['Kepler-9c']/periods['Kepler-9b']:.4f} (near 2:1)")
    print("  wrote kepler9_ttv.csv")


if __name__ == "__main__":
    main()
