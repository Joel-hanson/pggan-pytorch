import os
import sys
import copy
import time
import argparse
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.models as models

from io import BytesIO
from math import floor, ceil
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim import Adam
from torch.autograd import Variable
from torch.data import DataLoader
from torch.nn.init import kaiming_normal, calculate_gain
from torchvision import datasets
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter






# configuration
parser = argparse.ArgumentParser('PGGAN')

## general settings.
parser.add_argument('--train_data_root', type=str, default='/homes/user/Desktop/YOUR_DIRECTORY')
parser.add_argument('--random_seed', type=int, default=int(time.time()))
parser.add_argument('--n_gpu', type=int, default=1)             # for Multi-GPU training.

## training parameters.
parser.add_argument('--lr', type=float, default=0.001)          # learning rate.
parser.add_argument('--lr_decay', type=float, default=0.87)     # learning rate decay at every resolution transition.
parser.add_argument('--eps_drift', type=float, default=0.001)   # coeff for the drift loss.
parser.add_argument('--smoothing', type=float, default=0.997)   # smoothing factor for smoothed generator.
parser.add_argument('--nc', type=int, default=3)                # number of input channel.
parser.add_argument('--nz', type=int, default=512)              # input dimension of noise.
parser.add_argument('--ngf', type=int, default=512)             # feature dimension of final layer of generator.
parser.add_argument('--ndf', type=int, default=512)             # feature dimension of first layer of discriminator.
parser.add_argument('--TICK', type=int, default=1000)           # 1 tick = 1000 images = (1000/batch_size) iter.
parser.add_argument('--max_resl', type=int, default=8)          # 10-->1024, 9-->512, 8-->256
parser.add_argument('--trns_tick', type=int, default=200)       # transition tick
parser.add_argument('--stab_tick', type=int, default=100)       # stabilization tick


## network structure.
parser.add_argument('--flag_wn', type=bool, default=True)           # use of equalized-learning rate.
parser.add_argument('--flag_bn', type=bool, default=False)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_pixelwise', type=bool, default=True)    # use of pixelwise normalization for generator.
parser.add_argument('--flag_gdrop', type=bool, default=True)        # use of generalized dropout layer for discriminator.
parser.add_argument('--flag_leaky', type=bool, default=True)        # use of leaky relu instead of relu.
parser.add_argument('--flag_tanh', type=bool, default=False)        # use of tanh at the end of the generator.
parser.add_argument('--flag_sigmoid', type=bool, default=False)     # use of sigmoid at the end of the discriminator.
parser.add_argument('--flag_add_noise', type=bool, default=True)    # add noise to the real image(x)
parser.add_argument('--flag_norm_latent', type=bool, default=False) # pixelwise normalization of latent vector (z)
parser.add_argument('--flag_add_drift', type=bool, default=True)   # add drift loss


## optimizer setting.
parser.add_argument('--optimizer', type=str, default='adam')        # optimizer type.
parser.add_argument('--beta1', type=float, default=0.0)             # beta1 for adam.
parser.add_argument('--beta2', type=float, default=0.99)            # beta2 for adam.


## display and save setting.
parser.add_argument('--use_tb', type=bool, default=True)            # enable tensorboard visualization
parser.add_argument('--save_img_every', type=int, default=20)       # save images every specified iteration.
parser.add_argument('--display_tb_every', type=int, default=5)      # display progress every specified iteration.


## parse and save config.
config, _ = parser.parse_known_args()

# Utils
# import utils as utils

def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias = drange_out[0]-drange_in[0]*scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
        ])
    return transform(x)


def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


def save_image_single(x, path, imsize=512):
    from PIL import Image
    grid = make_image_grid(x, 1)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)


def save_image_grid(x, path, imsize=512, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)



def load_model(net, path):
    net.load_state_dict(torch.load(path))

def save_model(net, path):
    torch.save(net.state_dict(), path)


def make_summary(writer, key, value, step):
    if hasattr(value, '__len__'):
        for idx, img in enumerate(value):
            summary = tf.Summary()
            sio = BytesIO()
            scipy.misc.toimage(img).save(sio, format='png')
            image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
            summary.value.add(tag="{}/{}".format(key, idx), image=image_summary)
            writer.add_summary(summary, global_step=step)
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, global_step=step)


def mkdir(path):
    if os.name == 'nt':
        if not os.path.exists(path.replace('/', '\\')):
            os.makedirs(path.replace('/', '\\'))
    else:
        if not os.path.exists(path):
            os.makedirs(path)


irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


# Log
# import tf_recorder as tensorboard


class tf_recorder:
    def __init__(self):
        mkdir('repo/tensorboard')
        
        for i in range(1000):
            self.targ = 'repo/tensorboard/try_{}'.format(i)
            if not os.path.exists(self.targ):
                self.writer = SummaryWriter(self.targ)
                break
                
    def add_scalar(self, index, val, niter):
        self.writer.add_scalar(index, val, niter)

    def add_scalars(self, index, group_dict, niter):
        self.writer.add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter):
        grid = make_image_grid(x, ngrid)
        self.writer.add_image(index, grid, niter)

    def add_image_single(self, index, x, niter):
        self.writer.add_image(index, x, niter)

    def add_graph(self, index, x_input, model):
        torch.onnx.export(model, x_input, os.path.join(self.targ, "{}.proto".format(index)), verbose=True)
        self.writer.add_graph_onnx(os.path.join(self.targ, "{}.proto".format(index)))

    def export_json(self, out_file):
        self.writer.export_scalars_to_json(out_file)


# Dataloader
# import dataloader as DL



class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4
        
    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
        
        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))
        self.dataset = ImageFolder(
                    root=self.root,
                    transform=transforms.Compose(   [
                                                    transforms.Resize(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
                                                    transforms.ToTensor(),
                                                    ]))

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

       
    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]


# Layers
# from custom_layers import *

# same function as ConcatTable container in Torch7.
class ConcatTable(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        
    def forward(self,x):
        y = [self.layer1(x), self.layer2(x)]
        return y

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class fadein_layer(nn.Module):
    def __init__(self, config):
        super(fadein_layer, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, delta):
        self.alpha = self.alpha + delta
        self.alpha = max(0, min(self.alpha, 1.0))

    # input : [x_low, x_high] from ConcatTable()
    def forward(self, x):
        return torch.add(x[0].mul(1.0-self.alpha), x[1].mul(self.alpha))



# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class minibatch_std_concat_layer(nn.Module):
    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)


class pixelwise_norm_layer(nn.Module):
    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5


# for equaliaeed-learning rate.
class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.conv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':   xavier_normal(self.conv.weight)
        
        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1).expand_as(x)
        
 
class equalized_deconv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming'):
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.deconv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':   xavier_normal(self.deconv.weight)
        
        deconv_w = self.deconv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data/self.scale)
    def forward(self, x):
        x = self.deconv(x.mul(self.scale))
        return x + self.bias.view(1,-1,1,1).expand_as(x)


class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.linear.weight)
        
        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)
        
    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1,-1).expand_as(x)


# ref: https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class generalized_drop_out(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(generalized_drop_out, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str



# Network
# import network as net


# defined for code simplicity.
def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, only=False):
    if wn:  layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:          layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(equalized_linear(c_in, c_out))
    else:       layers.append(Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers

    
def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                          # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())         # copy weights
    return new_module

def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0-tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))

def get_module_names(model):
    names = []
    for key in model.state_dict().keys():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = config.nc
        self.nz = config.nz
        self.ngf = config.ngf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_gen()

    def first_block(self):
        layers = []
        ndim = self.ngf
        if self.flag_norm_latent:
            layers.append(pixelwise_norm_layer())
        layers = deconv(layers, self.nz, ndim, 4, 1, 3, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return  nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2, resl)), int(pow(2, resl)))
        ndim = self.ngf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ngf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))       # scale up by factor of 2.0
        if halving:
            layers = deconv(layers, ndim*2, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        else:
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
            layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return  nn.Sequential(*layers), ndim, layer_name
    
    def to_rgb_block(self, c_in):
        layers = []
        layers = deconv(layers, c_in, self.nc, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise, only=True)
        if self.flag_tanh:  layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model
    
    def grow_network(self, resl):
        # we make new network since pytorch does not support remove_module()
        new_model = nn.Sequential()
        names = get_module_names(self.model)
        for name, module in self.model.named_children():
            if not name=='to_rgb_block':
                new_model.add_module(name, module)                      # make new structure and,
                new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            
        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.Upsample(scale_factor=2, mode='nearest'))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))

            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)
           
    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_to_rgb')
           
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            # now, add the high resolution block.
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1, 1, 1))
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_sigmoid = config.flag_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def last_block(self):
        # add minibatch_std_concat_layer later.
        ndim = self.ndf
        layers = []
        layers.append(minibatch_std_concat_layer())
        layers = conv(layers, ndim+1, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = linear(layers, ndim, 1, sig=self.flag_sigmoid, wn=self.flag_wn)
        return  nn.Sequential(*layers), ndim
    
    def intermediate_block(self, resl):
        halving = False
        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2,resl)), int(pow(2,resl)), int(pow(2, resl-1)), int(pow(2, resl-1)))
        ndim = self.ndf
        if resl==3 or resl==4 or resl==5:
            halving = False
            ndim = self.ndf
        elif resl==6 or resl==7 or resl==8 or resl==9 or resl==10:
            halving = True
            for i in range(int(resl)-5):
                ndim = ndim/2
        ndim = int(ndim)
        layers = []
        if halving:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim*2, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        else:
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
            layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        
        layers.append(nn.AvgPool2d(kernel_size=2))       # scale up by factor of 2.0
        return  nn.Sequential(*layers), ndim, layer_name
    
    def from_rgb_block(self, ndim):
        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        return  nn.Sequential(*layers)
    
    def get_init_dis(self):
        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model
    

    def grow_network(self, resl):
            
        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2,resl-1)), int(pow(2,resl-1)), int(pow(2,resl)), int(pow(2,resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_downsample', nn.AvgPool2d(kernel_size=2))
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block(ndim))
            next_block.add_module('high_resl_block', inter_block)

            new_model = nn.Sequential()
            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))

            # we make new network since pytorch does not support remove_module()
            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name=='from_rgb_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_from_rgb')
           
            # add the high resolution block.
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)
            
            # add rest.
            for name, module in self.model.named_children():
                if name!='concat_block' and name!='fadein_block':
                    new_model.add_module(name, module)                      # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())      # copy pretrained weights

            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model
    
    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x



# Trainer

class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2           # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = {'gen':0, 'dis':0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # network and cirterion
        self.G = Generator(config).to(self.device)
        self.D = Discriminator(config).to(self.device)
        print ('Generator structure: ')
        print(self.G.model)
        print ('Discriminator structure: ')
        print(self.D.model)
        self.mse = torch.nn.MSELoss()
        if self.use_cuda:
            self.mse = self.mse.cuda()
            torch.cuda.manual_seed(config.random_seed)
            if config.n_gpu==1:
                self.G = torch.nn.DataParallel(self.G)
                self.D = torch.nn.DataParallel(self.D)
            else:
                gpus = []
                for i  in range(config.n_gpu):
                    gpus.append(i)
                self.G = torch.nn.DataParallel(self.G, device_ids=gpus).cuda()
                self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()  

        
        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()
        
        # tensorboard
        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tf_recorder()
        

    def resl_scheduler(self):
        '''
        this function will schedule image resolution(self.resl) progressively.
        it should be called every iteration to ensure resl value is updated properly.
        step 1. (trns_tick) --> transition in generator.
        step 2. (stab_tick) --> stabilize.
        step 3. (trns_tick) --> transition in discriminator.
        step 4. (stab_tick) --> stabilize.
        '''
        if floor(self.resl) != 2 :
            self.trns_tick = self.config.trns_tick
            self.stab_tick = self.config.stab_tick
        
        self.batchsize = self.loader.batchsize
        delta = 1.0/(2*self.trns_tick+2*self.stab_tick)
        d_alpha = 1.0*self.batchsize/self.trns_tick/self.TICK

        # update alpha if fade-in layer exist.
        if self.fadein['gen'] is not None:
            if self.resl%1.0 < (self.trns_tick)*delta:
                self.fadein['gen'].update_alpha(d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha*100
                self.phase = 'gtrns'
            elif self.resl%1.0 >= (self.trns_tick)*delta and self.resl%1.0 < (self.trns_tick+self.stab_tick)*delta:
                self.phase = 'gstab'
        if self.fadein['dis'] is not None:
            if self.resl%1.0 >= (self.trns_tick+self.stab_tick)*delta and self.resl%1.0 < (self.stab_tick + self.trns_tick*2)*delta:
                self.fadein['dis'].update_alpha(d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha*100
                self.phase = 'dtrns'
            elif self.resl%1.0 >= (self.stab_tick + self.trns_tick*2)*delta and self.phase!='final':
                self.phase = 'dstab'
            
        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        if (self.kimgs%self.TICK) < (prev_kimgs%self.TICK):
            self.globalTick = self.globalTick + 1
            # increase linearly every tick, and grow network structure.
            prev_resl = floor(self.resl)
            self.resl = self.resl + delta
            self.resl = max(2, min(10.5, self.resl))        # clamping, range: 4 ~ 1024

            # flush network.
            if self.flag_flush_gen and self.resl%1.0 >= (self.trns_tick+self.stab_tick)*delta and prev_resl!=2:
                if self.fadein['gen'] is not None:
                    self.fadein['gen'].update_alpha(d_alpha)
                    self.complete['gen'] = self.fadein['gen'].alpha*100
                self.flag_flush_gen = False
                self.G.module.flush_network()   # flush G
                print(self.G.module.model)
                #self.Gs.module.flush_network()         # flush Gs
                self.fadein['gen'] = None
                self.complete['gen'] = 0.0
                self.phase = 'dtrns'
            elif self.flag_flush_dis and floor(self.resl) != prev_resl and prev_resl!=2:
                if self.fadein['dis'] is not None:
                    self.fadein['dis'].update_alpha(d_alpha)
                    self.complete['dis'] = self.fadein['dis'].alpha*100
                self.flag_flush_dis = False
                self.D.module.flush_network()   # flush and,
                print(self.D.module.model)
                self.fadein['dis'] = None
                self.complete['dis'] = 0.0
                if floor(self.resl) < self.max_resl and self.phase != 'final':
                    self.phase = 'gtrns'

            # grow network.
            if floor(self.resl) != prev_resl and floor(self.resl)<self.max_resl+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.grow_network(floor(self.resl))
                #self.Gs.grow_network(floor(self.resl))
                self.D.grow_network(floor(self.resl))
                self.renew_everything()
                self.fadein['gen'] = dict(self.G.model.named_children())['fadein_block']
                self.fadein['dis'] = dict(self.D.model.named_children())['fadein_block']
                self.flag_flush_gen = True
                self.flag_flush_dis = True

            if floor(self.resl) >= self.max_resl and self.resl%1.0 >= (self.stab_tick + self.trns_tick*2)*delta:
                self.phase = 'final'
                self.resl = self.max_resl + (self.stab_tick + self.trns_tick*2)*delta


            
    def renew_everything(self):
        # renew dataloader.
        self.loader = dataloader(config)
        self.loader.renew(min(floor(self.resl), self.max_resl))
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0)
		
        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.z = Variable(self.z)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
        

    def feed_interpolated_input(self, x):
        if self.phase == 'gtrns' and floor(self.resl)>2 and floor(self.resl)<=self.max_resl:
            alpha = self.complete['gen']/100.0
            transform = transforms.Compose( [   transforms.ToPILImage(),
                                                transforms.Scale(size=int(pow(2,floor(self.resl)-1)), interpolation=0),      # 0: nearest
                                                transforms.Scale(size=int(pow(2,floor(self.resl))), interpolation=0),      # 0: nearest
                                                transforms.ToTensor(),
                                            ] )
            x_low = x.clone().add(1).mul(0.5)
            for i in range(x_low.size(0)):
                x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x = torch.add(x.mul(alpha), x_low.mul(1-alpha)) # interpolated_x

        if self.use_cuda:
            return x.cuda()
        else:
            return x

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z

    def train(self):
        # noise for test.
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz)
        if self.use_cuda:
            self.z_test = self.z_test.cuda()
        self.z_test = Variable(self.z_test, volatile=True)
        self.z_test.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
        
        for step in range(2, self.max_resl+1+5):
            for iter in tqdm(range(0,(self.trns_tick*2+self.stab_tick*2)*self.TICK, self.loader.batchsize)):
                self.globalIter = self.globalIter+1
                self.stack = self.stack + self.loader.batchsize
                if self.stack > ceil(len(self.loader.dataset)):
                    self.epoch = self.epoch + 1
                    self.stack = int(self.stack%(ceil(len(self.loader.dataset))))

                # reslolution scheduler.
                self.resl_scheduler()
                
                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                self.x.data = self.feed_interpolated_input(self.loader.get_batch())
                if self.flag_add_noise:
                    self.x = self.add_noise(self.x)
                self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
                self.x_tilde = self.G(self.z)
               
                self.fx = self.D(self.x)
                self.fx_tilde = self.D(self.x_tilde.detach())
                
                loss_d = self.mse(self.fx.squeeze(), self.real_label) + self.mse(self.fx_tilde, self.fake_label)
                loss_d.backward()
                self.opt_d.step()

                # update generator.
                fx_tilde = self.D(self.x_tilde)
                loss_g = self.mse(fx_tilde.squeeze(), self.real_label.detach())
                loss_g.backward()
                self.opt_g.step()
                
                # logging.
                log_msg = ' [E:{0}][T:{1}][{2:6}/{3:6}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{11:.5f}][cur:{6:.3f}][resl:{7:4}][{8}][{9:.1f}%][{10:.1f}%]'.format(self.epoch, self.globalTick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resl, int(pow(2,floor(self.resl))), self.phase, self.complete['gen'], self.complete['dis'], self.lr)
                tqdm.write(log_msg)

                # save model.
                self.snapshot('repo/model')

                # save image grid.
                if self.globalIter%self.config.save_img_every == 0:
                    with torch.no_grad():
                        x_test = self.G(self.z_test)
                    mkdir('repo/save/grid')
                    save_image_grid(x_test.data, 'repo/save/grid/{}_{}_G{}_D{}.jpg'.format(int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))
                    mkdir('repo/save/resl_{}'.format(int(floor(self.resl))))
                    save_image_single(x_test.data, 'repo/save/resl_{}/{}_{}_G{}_D{}.jpg'.format(int(floor(self.resl)),int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))

                # tensorboard visualization.
                if self.use_tb:
                    with torch.no_grad():
                        x_test = self.G(self.z_test)
                    self.tb.add_scalar('data/loss_g', loss_g.item(), self.globalIter)
                    self.tb.add_scalar('data/loss_d', loss_d.item(), self.globalIter)
                    self.tb.add_scalar('tick/lr', self.lr, self.globalIter)
                    self.tb.add_scalar('tick/cur_resl', int(pow(2,floor(self.resl))), self.globalIter)
                    '''IMAGE GRID
                    self.tb.add_image_grid('grid/x_test', 4, adjust_dyn_range(x_test.data.float(), [-1,1], [0,1]), self.globalIter)
                    self.tb.add_image_grid('grid/x_tilde', 4, adjust_dyn_range(self.x_tilde.data.float(), [-1,1], [0,1]), self.globalIter)
                    self.tb.add_image_grid('grid/x_intp', 4, adjust_dyn_range(self.x.data.float(), [-1,1], [0,1]), self.globalIter)
                    '''

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def snapshot(self, path):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stab phase.
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resl)), self.globalTick)
        if self.globalTick%50==0:
            if self.phase == 'gstab' or self.phase =='dstab' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))

# perform training

for k, v in vars(config).items():
    print('  {}: {}'.format(k, v))
print('-------------------------------------------------')
torch.backends.cudnn.benchmark = True           # boost speed.
trainer = trainer(config)
trainer.train()


