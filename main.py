import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import NeuralSDE
from datasets import SinData

# model save path
model_pth = "ckpts/ckpts.pth"

# training loop
def train(num_epochs, lr, batch_size, n_samples, t_size, state_size, bm_size):

    # initilize model
    model = NeuralSDE(sde_type="stratonovich",
                      noise_type="general",
                      state_size=state_size,
                      hidden_size=30,
                      bm_size=bm_size,
                      batch_size=batch_size)
    
    # load sin data
    sin_data = SinData(n_samples=n_samples, t_size=t_size)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_iterations = np.ceil(num_epochs/batch_size)
    # training loop
    for epoch in range(epochs):
        for i, (y_true, ts) in enumerate(sin_data): #simply iterating throuh whole dataset, no batching
            # forward pass
            y_pred = model(ts=ts, x=y_true)

            # backward pass
            loss = criterion(y_pred, y_true[3:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print progress
            if (i+1) % 5 == 0:
                torch.save(model.state_dict(), model_pth)
                print(f"epoch {epoch}/{epoch}, step {i+1}/{n_iterations}, Loss = {loss}")

# hyper parameters
epochs = 500
lr = 0.01
batch_size = 10
n_samples = 100
t_size = 30
state_size = 1
bm_size = 3

train(num_epochs=epochs,
      lr=lr,
      batch_size=batch_size,
      n_samples=n_samples,
      t_size=t_size,
      state_size=state_size,
      bm_size=bm_size)