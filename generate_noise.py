# Code based on https://github.com/tamarott/SinGAN
import torch


def generate_spatial_noise(size, device, embeddings=None, *args, **kwargs):
    """ Generates a noise tensor. Currently uses torch.randn. """
    # noise = generate_noise([size[0], *size[2:]], *args, **kwargs)
    # return noise.expand(size)
    if embeddings is not None:
        directions = torch.stack(embeddings.values())
        rand_directions = torch.randint(0, len(directions), size[0], *size[2:])
        
        # TODO(frederik): return randn amount in direction from current point to chosen embedding
        pass
    else:
        return torch.randn(size, device=device)

