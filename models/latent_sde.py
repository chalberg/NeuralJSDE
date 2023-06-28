import torch
import torch.nn as nn
import torchsde
from utils import _stable_division

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define neural SDE
class NeuralSDE(nn.Module):
    def __init__(self, noise_type, sde_type, state_size, hidden_size, bm_size, batch_size):
        super(NeuralSDE, self).__init__()

        self.noise_type = noise_type
        self.sde_type = sde_type
        self.state_size = state_size
        self.bm_size = bm_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.post_net = nn.Sequential(nn.Linear(state_size + 1, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, state_size))
        
        self.diffusion_net = nn.Sequential(nn.Linear(state_size + 1, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, state_size * bm_size))
        
        self.prior_net = nn.Sequential(nn.Linear(state_size+1, hidden_size),
                                       nn.Softmax(),
                                       nn.Linear(hidden_size, state_size))
        
        # simple encoder takes in fist 3 observations, outputs inital condition
        # TO DO: make into RNN to capture time dependencies
        self.encoder = nn.Sequential(nn.Linear(3, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, state_size))
    # appx posterior drift
    def f(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.post_net(tx)
    
    # prior drift
    def h(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.prior_net(tx)

    # diffusion
    def g(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.diffusion_net(tx).view(self.batch_size, self.state_size, self.bm_size)
    
    # augmented state drift (X, u)
    def f_aug(self, t, x):
        x = x[:, 0:1] 
        f, g, h = self.f(t, x), self.g(t, x), self.h(t, x)
        u = _stable_division(f - h, g)
        kl = .5 * (u ** 2).sum(dim=1, keepdim=True) # 1/2 ||u||_2^2 = KL(q||p)
        return torch.cat([f, kl], dim=1)
    
    # augmented state diffusion (X, u)
    def g_aug(self, t, x):
        x = x[:, 0:1]
        g = self.g(t, x)
        g_logqp = torch.zeros_like(x)
        return torch.cat([g, g_logqp], dim=1)
    
    def forward(self, ts, x):
        # Initial condition
        y0 = self.encoder(x[:3])  # shape (t_size)
        #Y = torch.zeros(len(ts), device=device) # shape (t_size)

        # generate Brownian motion sample
        bm = torchsde.BrownianInterval(t0=ts[3].item(),
                                       t1=ts[-1].item(),
                                       size=(self.state_size, self.bm_size),
                                       device=device)
            
        # numerically solve SDE
        y0 = y0.unsqueeze(0)
        sol = torchsde.sdeint_adjoint(sde=self,
                                       y0=y0,
                                       ts=ts[3:],
                                       bm=bm,
                                       method="midpoint",
                                       names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        sol = sol.squeeze().view(-1)
        return sol # [ys, kl]