"""
Normalizing-flow posterior head: a neural spline flow (via zuko) that is a DROP-IN replacement
for the MDN (model.py). It exposes the exact same interface, `.nll(x, theta)` and
`.sample(x, n)`, so train.py / evaluate.py can use either head by flipping one switch.

Why: the MDN's training becomes unstable spanning the resonance (val NLL diverges; only
early-stopping saves it). A flow optimizes a clean exact-likelihood objective and trains
stably, reaching comparable or better informativeness while staying calibrated (see
flow_head.py / flow_head_multiseed.py). It is for STABLE TRAINING + FAITHFUL, possibly
multimodal posteriors, not for magically sharper marginals (those are information-limited).
"""
import torch
import torch.nn as nn
import zuko


class NPEFlow(nn.Module):
    """Conditional neural spline flow q(theta | x). Same API as model.MDN."""

    def __init__(self, in_dim, theta_dim=6, transforms=5, hidden=(128, 128)):
        super().__init__()
        self.theta_dim = theta_dim
        self.flow = zuko.flows.NSF(features=theta_dim, context=in_dim,
                                   transforms=transforms, hidden_features=hidden)

    def nll(self, x, theta):
        """Negative log-likelihood of theta under q(theta | x). x:(B,in_dim) theta:(B,D)."""
        return -self.flow(x).log_prob(theta).mean()

    @torch.no_grad()
    def sample(self, x, n=2000):
        """Draw posterior samples. x:(B,in_dim) -> (B, n, D), matching MDN.sample."""
        s = self.flow(x).sample((n,))          # (n, B, D)
        return s.permute(1, 0, 2).contiguous()  # (B, n, D)
