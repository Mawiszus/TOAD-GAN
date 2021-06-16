# Contains code based on https://github.com/tamarott/SinGAN
import os
import torch

from mario.tokens import TOKEN_GROUPS

from .generator import Level_GeneratorConcatSkip2CleanAdd
from .discriminator import Level_WDiscriminator

import importlib
from models.base_model import BaseModel


def weights_init(m):
    """ Init weights for Conv and Norm Layers. """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Conv3d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_models(opt, use_softmax=True):
    """ Initialize Generator and Discriminator. """
    # generator initialization:
    G = Level_GeneratorConcatSkip2CleanAdd(opt, use_softmax).to(opt.device)
    G.apply(weights_init)
    if opt.netG != "":
        G.load_state_dict(torch.load(opt.netG))
    print(G)

    # discriminator initialization:
    D = Level_WDiscriminator(opt).to(opt.device)
    D.apply(weights_init)
    if opt.netD != "":
        D.load_state_dict(torch.load(opt.netD))
    print(D)

    return D, G


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def save_networks(G, D, z_opt, opt):
    torch.save(G.state_dict(), "%s/G.pth" % (opt.outf))
    torch.save(D.state_dict(), "%s/D.pth" % (opt.outf))
    torch.save(z_opt, "%s/z_opt.pth" % (opt.outf))


def restore_weights(D_curr, G_curr, scale_num, opt):
    G_state_dict = torch.load("%s/%d/G.pth" % (opt.out_, scale_num - 1))
    D_state_dict = torch.load("%s/%d/D.pth" % (opt.out_, scale_num - 1))

    G_head_conv_weight = G_state_dict["head.conv.weight"]
    G_state_dict["head.conv.weight"] = G_curr.head.conv.weight
    G_tail_weight = G_state_dict["tail.0.weight"]
    G_state_dict["tail.0.weight"] = G_curr.tail[0].weight
    G_tail_bias = G_state_dict["tail.0.bias"]
    G_state_dict["tail.0.bias"] = G_curr.tail[0].bias
    D_head_conv_weight = D_state_dict["head.conv.weight"]
    D_state_dict["head.conv.weight"] = D_curr.head.conv.weight

    for i, token in enumerate(opt.token_list):
        for group_idx, group in enumerate(TOKEN_GROUPS):
            if token in group:
                G_state_dict["head.conv.weight"][:, i] = G_head_conv_weight[:, group_idx]
                G_state_dict["tail.0.weight"][i] = G_tail_weight[group_idx]
                G_state_dict["tail.0.bias"][i] = G_tail_bias[group_idx]
                D_state_dict["head.conv.weight"][:, i] = D_head_conv_weight[:, group_idx]
                break

    G_state_dict["head.conv.weight"] = (G_state_dict["head.conv.weight"].detach().requires_grad_())
    G_state_dict["tail.0.weight"] = (G_state_dict["tail.0.weight"].detach().requires_grad_())
    G_state_dict["tail.0.bias"] = G_state_dict["tail.0.bias"].detach().requires_grad_()
    D_state_dict["head.conv.weight"] = (D_state_dict["head.conv.weight"].detach().requires_grad_())

    G_curr.load_state_dict(G_state_dict)
    D_curr.load_state_dict(D_state_dict)

    G_curr.head.conv.weight = torch.nn.Parameter(G_curr.head.conv.weight.detach().requires_grad_())
    G_curr.tail[0].weight = torch.nn.Parameter(G_curr.tail[0].weight.detach().requires_grad_())
    G_curr.tail[0].bias = torch.nn.Parameter(G_curr.tail[0].bias.detach().requires_grad_())
    D_curr.head.conv.weight = torch.nn.Parameter(D_curr.head.conv.weight.detach().requires_grad_())

    return D_curr, G_curr


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def load_trained_pyramid(opt):
    dir = opt.out_
    if(os.path.exists(dir)):
        reals = torch.load('%s/reals.pth' % dir)
        Gs = torch.load('%s/generators.pth' % dir)
        Zs = torch.load('%s/noise_maps.pth' % dir)
        NoiseAmp = torch.load('%s/noise_amplitudes.pth' % dir)

    else:
        print('no appropriate trained model exists, please train first')
    return Gs,Zs,reals,NoiseAmp


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model