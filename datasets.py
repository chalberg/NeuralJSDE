import torch
from torch.utils.data import Dataset

class SinData(Dataset):
    def __init__(self, t_size, n_samples):
        self.t_size = t_size
        self.n_samples = n_samples

        self.y, self.ts = self.make_sin_data(t_size, n_samples)
    
    def __getitem__(self, index):
        return self.y[index], self.ts[index]

    def __len__(self):
        return self.n_samples

    def make_sin_data(self, t_size, n_samples):
        # Generate sin data with irregular, noisy measurements
        sin_data = torch.zeros((n_samples, t_size))
        timestamps = torch.zeros((n_samples, t_size))

        print("Generating irregular sin data ...")
        for i in range(n_samples):
            ts, _ = torch.sort(torch.rand(t_size) * 10)  # Random timestamps between 0 and 10
            amplitude = torch.rand(1) * 1.5 + 0.5  # Random amplitude between 0.5 and 2.0
            frequency = torch.rand(1) * 1.5 + 0.5  # Random frequency between 0.5 and 2.0
            phase = torch.rand(1) * 2 * torch.pi  # Random phase between 0 and 2*pi

            sine_wave = amplitude * torch.sin(frequency * ts + phase)  # Generate sine function
            noise = torch.randn(t_size) * 0.1

            sin_data[i] = sine_wave + noise
            timestamps[i] = ts

        print("Done!")
        return sin_data, timestamps