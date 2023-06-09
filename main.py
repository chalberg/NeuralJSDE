import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from models import NeuralSDE
from datasets import SinData

# training loop
def train(num_epochs, lr, batch_size, n_samples, t_size, state_size, hidden_size, bm_size, dataset):
    # initilize model
    model = NeuralSDE(sde_type="stratonovich",
                      noise_type="general",
                      state_size=state_size,
                      hidden_size=hidden_size,
                      bm_size=bm_size,
                      batch_size=batch_size)
    
    # load data
    if dataset == "sin":
        data = SinData(n_samples=n_samples, t_size=t_size)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_iterations = np.ceil(num_epochs/batch_size)
    
    # training loop
    for epoch in range(num_epochs):
        for i, (y_true, ts) in enumerate(data): #simply iterating throuh whole dataset, no batching
            # forward pass
            y_pred = model(ts=ts, x=y_true)

            # backward pass
            loss = criterion(y_pred, y_true[3:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print progress
            if (i+1) % 5 == 0:
                torch.save(model.state_dict(), args.model_path)
                print(f"epoch {epoch}/{num_epochs}, step {i+1}/{n_iterations}, Loss = {loss}")
            
            # plot first sample and predictions for each epoch
            if i==1:
                plt.figure(figsize=(10,6))
                plt.plot(ts.detach().numpy(), y_true.detach().numpy(), 'bx', label='True')
                plt.plot(ts[3:].detach().numpy(), y_pred.detach().numpy(), 'b-', label='Predicted')
                plt.xlabel('t')
                plt.ylabel('Y(t)')
                plt.legend()

                save_path = os.path.join(args.img_path, f'output_plot_{epoch}.png')
                plt.savefig(save_path)
                plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--state_size', type=int, default=1, help="Dimension of latent variable Z(t)")
    parser.add_argument('--hidden_size', type=int, default=64, help="Size of hidden layers")
    parser.add_argument('--bm_size', type=int, default=3, help="Dimension of Brownian motion B(t)")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help='Step size of the optimizer')
    parser.add_argument('--model_path', type=str, default='output/ckpts/ckpts.pth', help='Path to save model state dict')
    parser.add_argument('--img_path', type=str, default='output/images/', help="Path for images to be saved to")

    # dataset args
    parser.add_argument('--dataset', type=str, default='sin', help="Dataset for to train on")
    parser.add_argument('--n_samples', type=int, default=300, help='Number of samples to generate for the dataset')
    parser.add_argument('--t_size', type=int, default=30, help='Number of timesteps to generate per sample')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    # training
    train(num_epochs=args.epochs,
          lr=args.lr,
          batch_size=args.batch_size,
          n_samples=args.n_samples,
          t_size=args.t_size,
          state_size=args.state_size,
          bm_size=args.bm_size,
          hidden_size=args.hidden_size,
          dataset=args.dataset)