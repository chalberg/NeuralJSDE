# Project: Implementation of a neural SDE using torchsde
# Author: Charlie Halberg
# Date: 05/31/2023

import torch
import torch.nn as nn
import torchsde

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

        self.drift_net = nn.Sequential(nn.Linear(state_size + 1, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, state_size))
        
        self.diffusion_net = nn.Sequential(nn.Linear(state_size + 1, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, state_size * bm_size))
        
        # simple encoder takes in fist 3 observations, outputs inital condition
        # TO DO: make into RNN to capture time dependencies
        self.encoder = nn.Sequential(nn.Linear(3, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, state_size))
    # drift
    def f(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.drift_net(tx)
    
    # diffusion
    def g(self, t, x):
        t = t.unsqueeze(0)
        x = x.squeeze(0)
        tx = torch.stack([t, x], dim=-1)
        return self.diffusion_net(tx).view(1, self.state_size, self.bm_size)
    
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
        y = y0.unsqueeze(0)
        sol = torchsde.sdeint_adjoint(sde=self,
                                       y0=y,
                                       ts=ts[3:],
                                       bm=bm,
                                       method="euler_heun")
        sol = sol.squeeze().view(-1)
        return sol