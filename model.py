"""
Mixture Density Network (MDN) = the simplest form of amortized neural posterior estimation.

Input : a TTV feature vector (O-C residuals of the first N transits).
Output: parameters of a K-component Gaussian mixture over theta = (m2, e2).

A mixture (not a single Gaussian) is essential: the TTV posterior is a curved,
sometimes multi-modal "valley" (the mass-eccentricity degeneracy). One Gaussian
cannot represent that; a mixture can. Trained by maximizing the likelihood the
mixture assigns to the true theta of each simulation -> it learns the posterior.
"""
import torch
import torch.nn as nn

class MDN(nn.Module):
    def __init__(self, in_dim, theta_dim=2, n_comp=24, hidden=256):
        super().__init__()
        self.theta_dim = theta_dim
        self.n_comp = n_comp
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, n_comp)              # mixture weights
        self.means = nn.Linear(hidden, n_comp*theta_dim)    # component means
        self.logstd = nn.Linear(hidden, n_comp*theta_dim)   # component log-stds (diagonal)

    def forward(self, x):
        h = self.body(x)
        log_w = torch.log_softmax(self.logits(h), dim=-1)               # (B,K)
        mu = self.means(h).view(-1, self.n_comp, self.theta_dim)        # (B,K,D)
        log_sigma = self.logstd(h).view(-1, self.n_comp, self.theta_dim)
        log_sigma = torch.clamp(log_sigma, -7.0, 3.0)
        return log_w, mu, log_sigma

    def nll(self, x, theta):
        """Negative log-likelihood of true theta under the predicted mixture."""
        log_w, mu, log_sigma = self.forward(x)
        theta = theta.unsqueeze(1)                                      # (B,1,D)
        var = torch.exp(2*log_sigma)
        # log N(theta; mu, sigma) per component, summed over dims
        log_comp = -0.5*(((theta-mu)**2)/var + 2*log_sigma + torch.log(torch.tensor(2*torch.pi))).sum(-1)
        log_prob = torch.logsumexp(log_w + log_comp, dim=-1)            # (B,)
        return -log_prob.mean()

    @torch.no_grad()
    def sample(self, x, n=2000):
        """Draw posterior samples theta ~ p(theta | x). x:(B,in_dim) -> (B,n,D)."""
        log_w, mu, log_sigma = self.forward(x)
        B = x.shape[0]
        comp = torch.multinomial(log_w.exp(), n, replacement=True)      # (B,n)
        idx = comp.unsqueeze(-1).expand(-1, -1, self.theta_dim)         # (B,n,D)
        chosen_mu = torch.gather(mu, 1, idx)
        chosen_sig = torch.gather(torch.exp(log_sigma), 1, idx)
        eps = torch.randn(B, n, self.theta_dim, device=x.device)
        return chosen_mu + chosen_sig*eps
