import random

import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def prepare_mnist_seed_images():
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('output/mnist/', train=False, download=True,
                                   transform=torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=1, shuffle=True)
    eights = torch.zeros((20, 1, 28, 28))
    e = 0
    while e < eights.shape[0]:
        batch = next(iter(test_loader))
        if batch[1].item() == 8:
            eights[e] = batch[0]
            e += 1

    for i in range(len(eights)):
        tmp = eights[i, 0]
        x, y = torch.where(tmp > 0)
        l_x = max(x) - min(x)
        l_y = max(y) - min(y)
        if l_x == l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            y_1 = min(y)
            y_2 = max(y) + 2
        elif l_x > l_y:
            x_1 = min(x)
            x_2 = max(x) + 2
            diff = l_x - l_y
            y_1 = min(y) - diff//2
            y_2 = max(y) + diff//2 + 2
        else:  # l_y > l_x:
            y_1 = min(y)
            y_2 = max(y) + 2
            diff = l_y - l_x
            x_1 = min(x) - diff//2
            x_2 = max(x) + diff//2 + 2
        tmp = tmp[x_1:x_2, y_1:y_2]
        # tmp = interpolate(tmp.unsqueeze(0).unsqueeze(0), (28, 28))
        plt.imsave('mariokart/seed_road/MNIST_examples/eights/sample_%d.png' % i, tmp[0][0], cmap='Greys')
