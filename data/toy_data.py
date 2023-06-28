import torch
from torch.utils.data import Dataset
from torch.distributions.exponential import Exponential

class InformativeSinData(Dataset):
    def __init__(self, t_size, n_samples):
        self.t_size = t_size
        self.n_samples = n_samples

        self.y, self.ts = self.make_sin_data(t_size, n_samples)

    def __getitem__(self, index):
        return self.y[index], self.ts[index]

    def __len__(self):
        return self.n_samples

    def make_sin_data(self, t_size, n_samples):

        sin_data = torch.zeros((n_samples, t_size))
        timestamps = torch.zeros((n_samples, t_size))

        print("Generating informative sin data ...")
        for i in range(n_samples):

            ts = torch.zeros(t_size)
            sin_wave = torch.zeros(t_size)

            # sin wave params
            amplitude = torch.rand(1) * 1.5 + 0.5
            frequency = torch.rand(1) * 0.5 + 1
            phase = torch.rand(1) * 2 * torch.pi

            # initial conditions
            ts[0] = torch.rand(1)
            sin_wave[0] = (amplitude * torch.sin(frequency * ts[0] + phase))
            sin_wave[0] = sin_wave[0] + torch.randn(1) * 0.1

            # iteratively sample timestamps and values
            for j in range(1, t_size):
                rate = torch.abs(torch.cos(ts[j - 1])) + torch.exp(-sin_wave[j-1])  # rate param dependent on previous timestamp and value
                dt = Exponential(rate).sample()  # sample time step
                ts[j] = ts[j - 1] + dt
                sin_wave[j] = (amplitude * torch.sin(frequency * ts[j] + phase))
                sin_wave[j] = sin_wave[j] + torch.randn(1) * 0.1 # add gaussian noise
    
            timestamps[i] = ts
            sin_data[i] = sin_wave

        print("Done!")
        return sin_data, timestamps
