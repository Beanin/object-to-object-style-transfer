import os
import sys
import torch
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from .fcn import fcn8s
from .utils import convert_state_dict

def load_model(model_path):
    # Setup Model
    model = fcn8s(n_classes=21)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    state = convert_state_dict(torch.load(model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    
    model.cuda(0)
    return model

def get_softmask(model, img_path):

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    img = misc.imread(img_path)

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= [104.00699, 116.66877, 122.67892]
    img = misc.imresize(img, (512, 512))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    
    images = Variable(img.cuda(0), volatile=True)

    outputs = F.softmax(model(images), dim=1)
    outputs = outputs.cpu().data.numpy()
    outputs = outputs[0, :, :, :]

    return outputs
