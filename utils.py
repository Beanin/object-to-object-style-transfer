import torch
import itertools
import numpy as np
import skimage.io as io

from torch.nn.functional import conv3d
from torch.autograd import Variable

from torchvision import transforms, utils


def load_img(path):
    I = io.imread(path)
    return np.moveaxis(I, -1, 0)


def to_img(tensor):
    """ stolen from utils.save_image"""
    grid = utils.make_grid(tensor)
    return grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def get_patch(x, j, k, T):
    return x[0, :, j: T + j, k : k + T]

def extract_patches(x, newH, newW, T):
    return torch.cat([get_patch(x, i, j, T).unsqueeze(0) for i, j in itertools.product(range(newH), range(newW))], dim=0)


def preprocess_masks(content_masks, style_masks, H, W, T):
    newH, newW = H - T + 1, W - T + 1
    
    style_patches = extract_patches(style_masks, newH, newW, T)
    
    norm = (style_patches ** 2).view(style_patches.shape[0], -1).sum(1)

    return conv3d(content_masks.unsqueeze(0), style_patches.unsqueeze(1)), norm

def adjust_lr(optimizer, init_lr, decay, epoch):
    lr = init_lr * (decay ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
                               
def extract_layer_masks(initial_masks, h, w, beta, use_cuda=True):
    if torch.cuda.is_available and use_cuda:
        return Variable(beta * downsample(torch.FloatTensor(initial_masks), (h, w)).cuda(), requires_grad=False)
    else:
        return Variable(beta * downsample(torch.FloatTensor(initial_masks), (h, w)), requires_grad=False)

def downsample(imgs, sz):
    """Downsample a sequence of binary images to a given size."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(sz),
        transforms.ToTensor(),
    ])
    
    return torch.cat([transform(x.view((1, *x.shape))).view((1, 1, *sz)) for x in imgs], dim=1)

def prepare_img(img, requires_grad=False, use_cuda=True, resize=None):
    if not torch.is_tensor(img):
        tens = torch.from_numpy(img) # PIL Image inverts tensor for some reason
    else:
        tens = img
    
    if resize is not None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

        tens = transform(tens)

    if torch.cuda.is_available() and use_cuda:
        return Variable(tens.unsqueeze(0).cuda(), requires_grad=requires_grad)
    else:
        return Variable(tens.unsqueeze(0), requires_grad=requires_grad)