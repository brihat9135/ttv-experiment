"""
Does masked self-supervised pretraining on UNLABELED 'real-like' data close the
sim-to-real gap?  (Tests the transformer-autoencoder / embedding idea.)

Controlled synthetic gap (so we can SCORE accuracy — real data has no labels):
  IDEALIZED  domain: simulated TTVs + clean iid Gaussian noise
  REAL-LIKE  domain: simulated TTVs + correlated (AR1) red noise + heavy-tailed outliers

Pipeline: a small Transformer ENCODER over per-transit tokens -> pooled embedding ->
MDN posterior head over theta=(m1,m2,h1,k1,h2,k2). Arms:
  1 Reference        train ideal-labeled,            test IDEAL     (best case, no gap)
  2 Baseline (gap)   train ideal-labeled,            test REAL-LIKE (the sim-to-real gap)
  3 Pretrained-froze MAE-pretrain enc on REAL-LIKE-UNLABELED, freeze, head on ideal-lab, test REAL
  4 Pretrained-ft    same pretrain, fine-tune all on ideal-lab,  test REAL
  5 Mix              train on ideal-lab + REAL-LABELED mixed,    test REAL  (your 'mix' idea*)
  6 Oracle           train REAL-LABELED,                          test REAL (upper bound*)
  (* arms 5-6 use real-like LABELS, which real data does NOT have; shown as references.)

Pretraining helps iff arms 3/4 beat arm 2 (lower m2 RMSE / calErr), approaching 5/6.
"""
import time, copy, numpy as np, torch, torch.nn as nn
import simulator as S
from model import MDN

S.E1_MAX = S.E2_MAX = 0.12         # informative fixed-ratio regime (P2=21d, ratio 2.1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

N_LAB, N_UNLAB, N_TEST = 14000, 20000, 3000
D_MODEL, HEADS, LAYERS = 64, 4, 3
SEQ, N1 = S.FEATURE_DIM, S.N_TRANSITS_1

# per-token metadata: planet flag + within-planet normalized index
_flag = np.zeros(SEQ, np.float32); _flag[N1:] = 1.0
_idx = np.zeros(SEQ, np.float32)
_idx[:N1] = np.arange(N1)/(N1-1); _idx[N1:] = np.arange(SEQ-N1)/(SEQ-N1-1)
META = np.stack([_flag, _idx], 1)


def gen(n, seed):
    rng = np.random.default_rng(seed)
    th = S.sample_prior(n, rng)
    f = np.full((n, SEQ), np.nan)
    for i in range(0, n, 2500):
        sl = slice(i, min(i+2500, n)); f[sl] = S.simulate(th[sl])
    ok = ~np.isnan(f).any(1)
    return th[ok], f[ok]


def noise_ideal(f, rng):
    return f + rng.normal(0, 0.5, f.shape)

def noise_real(f, rng):
    out = f.copy()
    for a, b in [(0, N1), (N1, SEQ)]:           # AR(1) red noise within each planet block
        phi, sig = 0.7, 0.45
        e = rng.normal(0, sig, (f.shape[0], b-a)); red = np.zeros_like(e)
        red[:, 0] = e[:, 0]
        for t in range(1, b-a):
            red[:, t] = phi*red[:, t-1] + e[:, t]
        out[:, a:b] += red
    out += rng.normal(0, 0.25, f.shape)          # white floor
    out += (rng.random(f.shape) < 0.05) * rng.normal(0, 3.0, f.shape)   # heavy-tailed outliers
    return out


class Encoder(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(3, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d)*0.02)
        self.mask_tok = nn.Parameter(torch.randn(d)*0.02)
        layer = nn.TransformerEncoderLayer(d, HEADS, dim_feedforward=128, dropout=0.1,
                                           batch_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, LAYERS)
        self.register_buffer("meta", torch.tensor(META))

    def forward(self, values, mask=None):
        B = values.shape[0]
        meta = self.meta.unsqueeze(0).expand(B, -1, -1)
        emb = self.proj(torch.cat([values.unsqueeze(-1), meta], -1))   # (B,SEQ,d)
        if mask is not None:
            emb = torch.where(mask.unsqueeze(-1), self.mask_tok.view(1, 1, -1), emb)
        emb = emb + self.pos.unsqueeze(0)
        h = self.tf(emb)
        return h, h.mean(1)


class Posterior(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc; self.mdn = MDN(in_dim=D_MODEL, theta_dim=6)
    def nll(self, v, th):
        _, p = self.enc(v); return self.mdn.nll(p, th)
    def sample(self, v, n=600):
        _, p = self.enc(v); return self.mdn.sample(p, n)


def pretrain(enc, feats_unlab, scale, epochs=45):
    head = nn.Linear(D_MODEL, 1).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), 1e-3)
    X = torch.tensor(feats_unlab/scale, dtype=torch.float32)
    g = torch.Generator().manual_seed(5)
    for ep in range(epochs):
        enc.train(); perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            xb = X[perm[i:i+256]].to(DEVICE)
            m = torch.rand(xb.shape, device=DEVICE) < 0.15
            if m.sum() == 0: continue
            h, _ = enc(xb, mask=m)
            pred = head(h).squeeze(-1)
            loss = ((pred[m] - xb[m])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return enc


def train_down(enc, lab_th, lab_f, noise_fn, scale, th_m, th_s, freeze=False, epochs=120, lr=1e-3, seed=7):
    net = Posterior(enc).to(DEVICE)
    params = list(net.mdn.parameters()) + ([] if freeze else list(net.enc.parameters()))
    opt = torch.optim.Adam(params, lr)
    Y = torch.tensor((lab_th - th_m)/th_s, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        if freeze: net.enc.eval()
        X = torch.tensor(noise_fn(lab_f, rng)/scale, dtype=torch.float32)
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            ii = perm[i:i+256]; xb = X[ii].to(DEVICE); yb = Y[ii].to(DEVICE)
            if freeze:
                with torch.no_grad(): _, p = net.enc(xb)
                loss = net.mdn.nll(p, yb)
            else:
                loss = net.nll(xb, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    return net


def evaluate(net, te_th, te_f, noise_fn, scale, th_m, th_s):
    rng = np.random.default_rng(11)
    X = torch.tensor(noise_fn(te_f, rng)/scale, dtype=torch.float32).to(DEVICE)
    net.eval()
    with torch.no_grad():
        s = net.sample(X, n=600).cpu().numpy()*th_s + th_m
    m2err = float(np.sqrt(np.mean((s[:, :, 1].mean(1) - te_th[:, 1])**2)))
    cerr = []
    for lv in [0.5, 0.68, 0.9]:
        lo, hi = (1-lv)/2*100, (1+lv)/2*100
        for d in range(6):
            ql = np.percentile(s[:, :, d], lo, 1); qh = np.percentile(s[:, :, d], hi, 1)
            cerr.append(abs(np.mean((te_th[:, d] >= ql) & (te_th[:, d] <= qh)) - lv))
    return m2err, float(np.mean(cerr))


def noise_mix(f, rng):
    h = len(f)//2
    return np.concatenate([noise_ideal(f[:h], rng), noise_real(f[h:], rng)])


def main():
    print("Generating clean datasets...")
    t0 = time.time()
    lab_th, lab_f = gen(N_LAB, 0)
    unlab_th, unlab_f = gen(N_UNLAB, 1)
    test_th, test_f = gen(N_TEST, 2)
    print(f"  {time.time()-t0:.0f}s  (lab {len(lab_th)}, unlab {len(unlab_th)}, test {len(test_th)})")

    scale = float(lab_f.std()) + 1e-8
    th_m, th_s = lab_th.mean(0), lab_th.std(0) + 1e-8
    unlab_real = noise_real(unlab_f, np.random.default_rng(20))     # unlabeled real-like for pretraining

    print("Pretraining encoder (masked autoencoding on real-like UNLABELED)...")
    t0 = time.time()
    enc_pt = pretrain(Encoder().to(DEVICE), unlab_real, scale)
    print(f"  pretrained in {time.time()-t0:.0f}s")

    print("Training arms...")
    rows = []
    net_base = train_down(Encoder().to(DEVICE), lab_th, lab_f, noise_ideal, scale, th_m, th_s)
    rows.append(("1 Reference  (ideal->ideal)",) + evaluate(net_base, test_th, test_f, noise_ideal, scale, th_m, th_s))
    rows.append(("2 Baseline   (ideal->REAL) [gap]",) + evaluate(net_base, test_th, test_f, noise_real, scale, th_m, th_s))
    net_fz = train_down(copy.deepcopy(enc_pt), lab_th, lab_f, noise_ideal, scale, th_m, th_s, freeze=True)
    rows.append(("3 Pretrained-frozen  ->REAL",) + evaluate(net_fz, test_th, test_f, noise_real, scale, th_m, th_s))
    net_ft = train_down(copy.deepcopy(enc_pt), lab_th, lab_f, noise_ideal, scale, th_m, th_s, freeze=False, lr=3e-4)
    rows.append(("4 Pretrained-finetune ->REAL",) + evaluate(net_ft, test_th, test_f, noise_real, scale, th_m, th_s))
    mix_th = np.concatenate([lab_th, lab_th]); mix_f = np.concatenate([lab_f, lab_f])
    net_mix = train_down(Encoder().to(DEVICE), mix_th, mix_f, noise_mix, scale, th_m, th_s)
    rows.append(("5 Mix ideal+REAL-labeled ->REAL*",) + evaluate(net_mix, test_th, test_f, noise_real, scale, th_m, th_s))
    net_or = train_down(Encoder().to(DEVICE), lab_th, lab_f, noise_real, scale, th_m, th_s)
    rows.append(("6 Oracle  (REAL-labeled->REAL)*",) + evaluate(net_or, test_th, test_f, noise_real, scale, th_m, th_s))

    print("\n" + "="*70)
    print("DOES MASKED PRETRAINING CLOSE THE SIM-TO-REAL GAP?")
    print("="*70)
    print(f"{'arm':>34} {'m2 RMSE (Me)':>13} {'calErr':>8}")
    print("-"*70)
    for name, m2, ce in rows:
        print(f"{name:>34} {m2:>13.2f} {ce*100:>7.1f}%")
    print("-"*70)
    ref, base = rows[0][1], rows[1][1]
    print(f"sim-to-real GAP (m2 RMSE): reference {ref:.2f} -> baseline {base:.2f}  (+{base-ref:.2f})")
    print("Pretraining helps iff arms 3/4 sit below arm 2 and move toward arms 5/6.")
    print("(* arms 5-6 use real-like LABELS that real data does not have — references only.)")


if __name__ == "__main__":
    main()
