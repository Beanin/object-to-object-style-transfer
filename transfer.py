import skimage.io as io
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
from torch.nn.functional import conv3d

from torchvision import transforms
from torchvision.models import vgg19_bn

from segmentation.softmask import load_model, get_softmask
from utils import *

class VGGExtractor(nn.Module):
        def __init__(self, use_cuda=True):
            super().__init__()
            model = vgg19_bn(pretrained=True)
            print(model.features)
            if torch.cuda.is_available() and use_cuda:
                model = model.cuda()

            self.first = nn.Sequential(*[model.features[i] for i in range(10)])
            self.second = nn.Sequential(*[model.features[i] for i in range(10, 17)])
            self.third = nn.Sequential(*[model.features[i] for i in range(17, 30)])
            self.fourth = nn.Sequential(*[model.features[i] for i in range(30, 43)])

            for param in model.parameters():
                param.requires_grad = False

        def forward(self, x):
            conv2_1 = self.first.forward(x)
            conv3_1 = self.second.forward(conv2_1)
            conv4_1 = self.third.forward(conv3_1)
            conv5_1 = self.fourth.forward(conv4_1)
            return conv2_1, conv3_1, conv4_1, conv5_1

class StyleTransfer(object):
    def __init__(self, vgg=None):
        if vgg is None:
            vgg = VGGExtractor()
        self.nn = vgg
        print(self.nn.first)
        print(self.nn.second)
        print(self.nn.third)
        print(self.nn.fourth)
        
    def get_style_loss(self, content_masks, style_masks, gen_layer, style_layer, masks_product, masks_norm, T):
        EPS = 1e-8 # constant for numerical stability 
        
        _, _, H, W = style_layer[0].shape
        newH, newW = H - T + 1, W - T + 1

        p_s = torch.cat([extract_patches(layer, newH, newW, T) for layer in style_layer], dim=0)
        norm = ((p_s ** 2).view(p_s.shape[0], -1).sum() + masks_norm) ** 0.5

        norm = norm.unsqueeze(0)
        for i in range(len(masks_product.shape) - 2):
            norm = norm.unsqueeze(-1)

        feature_product = conv3d(gen_layer.unsqueeze(0), p_s.unsqueeze(1))

        NN = torch.max((feature_product + masks_product) / (norm + EPS), 1)[1]

        s = [torch.cat([layer, masks], dim=1) for layer, masks in zip(style_layer, style_masks)]
        p_s = torch.cat([extract_patches(x, newH, newW, T) for x in s], dim=0)

        g = torch.cat((gen_layer, content_masks), dim=1)
        p_g = extract_patches(g, newH, newW, T)
        
        #print(gen_layer.shape, style_layer.shape)
        #print("NN" ,NN.shape)
        #print("p_s, p_g", p_s.shape, p_g.shape)
        p_s = p_s[NN.view(-1)]
        
        
        
        if self.i % 100 == 0:
            self.debug['NN'].append(NN.data.cpu())
        # self.debug['feature_product'].append(feature_product.data.cpu())
        
        return ((p_g - p_s) ** 2).mean()
    
        
    def generate(self, gen_img, content, style, content_masks, style_masks, lr, decay, n_iter, sz=None,
                 T=3, ALPHA_1=1e2, ALPHA_2=1e2, ALPHA_3=5e2, BETA=8):
        if sz is None:
            h, w = content.shape[-2:]
        else:
            h, w = sz
        
        # Initialize generated image
        
        if gen_img is None:
            gen_img = np.random.randint(0, 255, content.shape, dtype=np.uint8)
            gen_var = prepare_img(gen_img, requires_grad=True, resize=(h, w))
        else:
            gen_var = prepare_img(gen_img, requires_grad=True, resize=(h, w))

        cont_var = prepare_img(content, resize=(h, w))
        style_var = [prepare_img(style_img, resize=(h, w)) for style_img in style]
        
        style2_1, style3_1, style4_1, style5_1 =  [], [], [], []
        for s2, s3, s4, s5 in map(self.nn.forward, style_var):
            style2_1.append(s2)
            style3_1.append(s3)
            style4_1.append(s4)
            style5_1.append(s5)
            
        cont2_1, cont3_1, cont4_1, cont5_1 = self.nn.forward(cont_var)
        
        
        # produce masks for augmented layers
        
        content_masks_0 = extract_layer_masks(content_masks, h, w, BETA)
        
        h, w = style3_1[0].shape[2:]

        content_masks_3_1 = extract_layer_masks(content_masks, h, w, BETA)
        style_masks_3_1 = [extract_layer_masks(msq, h, w, BETA) for msq in style_masks]
        
        masks_prod_3_1, masks_norm_3_1 = preprocess_masks(content_masks_3_1, style_masks_3_1, h, w, T)

        h, w = style4_1[0].shape[2:]
        content_masks_4_1 = extract_layer_masks(content_masks, h, w, BETA)
        style_masks_4_1 = [extract_layer_masks(msq, h, w, BETA) for msq in style_masks]
        masks_prod_4_1, masks_norm_4_1 = preprocess_masks(content_masks_4_1, style_masks_4_1, h, w, T)

        # initialize optimizer
        
        optimizer = optim.Adam([gen_var])
        
        ### initialize debug info
        
        self.debug = {}
        self.debug['loss'] = []
        self.debug['content_loss'] = []
        self.debug['style_loss_3_1'] = []
        self.debug['style_loss_4_1'] = []
        #self.debug['NN'] = []
        #self.debug['feature_product'] = []
        self.debug['mask_product'] = [masks_prod_3_1.data.cpu(), masks_prod_4_1.data.cpu()]
        self.debug['NN'] = []

        for self.i in range(n_iter):
            i = self.i
            
            adjust_lr(optimizer, lr, decay, i)

            optimizer.zero_grad()
            
            gen2_1, gen3_1, gen4_1, gen5_1 = self.nn.forward(gen_var)

            style_loss_3_1 = self.get_style_loss(content_masks_3_1, style_masks_3_1, gen3_1, style3_1,
                                            masks_prod_3_1, masks_norm_3_1, T)
            style_loss_4_1 = self.get_style_loss(content_masks_4_1, style_masks_4_1, gen4_1, style4_1,
                                            masks_prod_4_1, masks_norm_4_1, T)

            content_loss = ((cont5_1 - gen5_1) ** 2).mean() + ((cont4_1 - gen4_1) ** 2).mean()

            background_loss = ((content_masks_0[0, 0] * (cont_var - gen_var)) ** 2).mean()

            loss = ALPHA_1 * content_loss + ALPHA_2 * (style_loss_4_1 + style_loss_3_1) + ALPHA_3 * background_loss

            self.debug['loss'].append(loss.data[0])
            self.debug['content_loss'].append(content_loss.data[0])
            self.debug['style_loss_3_1'].append(style_loss_3_1.data[0])
            self.debug['style_loss_4_1'].append(style_loss_4_1.data[0])
            
            
            
            if i % 200 == 0:
                print("Iteration:", i, "Loss:", loss.data[0])
                print("Content Loss:", content_loss.data[0]* ALPHA_1)
                print("Style Loss 3_1:", style_loss_3_1.data[0] * ALPHA_2)
                print("Style Loss 4_1:", style_loss_4_1.data[0] * ALPHA_2)
                print("Background Loss:", background_loss.data[0] * ALPHA_3)
                
            loss.backward()
            
            optimizer.step()
            

        return gen_var.data.cpu()
        
class ObjectStyleTransfer(object):
    def __init__(self, segm_model_path):
        self._segmentation = load_model(segm_model_path)
        self._style_transfer = StyleTransfer()

    
    def generate(self, content_img, style_img, size=None, lr=1e-2,
                 T=3, ALPHA_1=1e2, ALPHA_2=1e2, ALPHA_3=5e2, BETA=20):
        content = load_img(content_img)
        style = [load_img(img) for img in style_img]
        
        content_masks = get_softmask(self._segmentation, content_img)
        style_masks = [get_softmask(self._segmentation, img) for img in style_img]
        
        gen_img = None
        h, w = (64, 64)
        
        if size is None:
            c_h, c_w = content.shape[-2:]
        else:
            c_h, c_w = size
            
        while h < c_h and w < c_w: 
            print((h, w))
            gen_img = self._style_transfer.generate(
                gen_img=None,
                content=content,
                style=style,
                content_masks=content_masks,
                style_masks=style_masks,
                lr=lr,
                decay=0.8,
                n_iter=201,
                sz=(h, w),
                T=T, 
                ALPHA_1=ALPHA_1, 
                ALPHA_2=ALPHA_2, 
                ALPHA_3=ALPHA_3, 
                BETA=BETA
            )
            h *= 2
            w *= 2
        
        print(c_h, c_w)
        gen_img = self._style_transfer.generate(
            gen_img=None,
            content=content,
            style=style,
            content_masks=content_masks,
            style_masks=style_masks,
            lr=lr,
            decay=0.8,
            n_iter=201,
            sz=(c_h, c_w),
            T=T, 
            ALPHA_1=ALPHA_1, 
            ALPHA_2=ALPHA_2, 
            ALPHA_3=ALPHA_3, 
            BETA=BETA
        )
        return to_img(gen_img)
