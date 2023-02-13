import numpy as np
import torch

import torchvision.transforms as Transforms

def gaussian_noise(x, scale=0.8):
    gaussian_data_x = x + np.random.normal(
        loc=0,
        scale=scale,
        size=x.shape
    )

    gaussian_data_x = np.clip(
        gaussian_data_x, 0, 1
    )

    gaussian_data_x = torch.tensor(gaussian_data_x)
    gaussian_data_x = gaussian_data_x.type(torch.FloatTensor)
    return gaussian_data_x

    