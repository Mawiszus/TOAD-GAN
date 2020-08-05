# Code based on https://github.com/tamarott/SinGAN
from torch.nn.functional import interpolate

from generate_noise import generate_spatial_noise
from mario.level_utils import group_to_token


def format_and_use_generator(curr_img, G_z, count, mode, Z_opt, pad_noise, pad_image, noise_amp, G, opt):
    """ Correctly formats input for generator and runs it through. """
    if curr_img.shape != G_z.shape:
        G_z = interpolate(G_z, curr_img.shape[-2:], mode='bilinear', align_corners=False)
    if count == (opt.token_insert + 1):  # (opt.stop_scale - 1):
        G_z = group_to_token(G_z, opt.token_list)
    if mode == "rand":
        curr_img = pad_noise(curr_img)  # Curr image is z in this case
        z_add = curr_img
    else:
        z_add = Z_opt
    G_z = pad_image(G_z)
    z_in = noise_amp * z_add + G_z
    G_z = G(z_in.detach(), G_z)
    return G_z


def draw_concat(generators, noise_maps, reals, noise_amplitudes, in_s, mode, pad_noise, pad_image, opt):
    """ Draw and concatenate output of the previous scale and a new noise map. """
    G_z = in_s
    if len(generators) > 0:
        if mode == "rand":
            noise_padding = 1 * opt.num_layer
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                if count < opt.stop_scale:  # - 1):
                    z = generate_spatial_noise([1,
                                                real_curr.shape[1],
                                                Z_opt.shape[2] - 2 * noise_padding,
                                                Z_opt.shape[3] - 2 * noise_padding],
                                               device=opt.device)
                G_z = format_and_use_generator(z, G_z, count, "rand", Z_opt,
                                               pad_noise, pad_image, noise_amp, G, opt)

        if mode == "rec":
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                    zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):
                G_z = format_and_use_generator(real_curr, G_z, count, "rec", Z_opt,
                                               pad_noise, pad_image, noise_amp, G, opt)

    return G_z
