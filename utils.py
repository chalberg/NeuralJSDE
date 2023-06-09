from torch import load
from models import NeuralSDE

def load_model(model_path, state_size, hidden_size, bm_size, batch_size):
    # Create an instance of NeuralSDE
    model = NeuralSDE(sde_type="stratonovich",
                      noise_type="general",
                      state_size=state_size,
                      hidden_size=hidden_size,
                      bm_size=bm_size,
                      batch_size=batch_size)

    # Load existing state dict
    model.load_state_dict(load(model_path))

    return model.eval()