import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from getter import get_model, get_loader, get_data_path
from metrics import runningScore
from loss import *
from augmentations import *

def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv_block4.parameters():
        param.requires_grad = True
    for param in model.conv_block5.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.score_pool4.parameters():
        param.requires_grad = True
    for param in model.score_pool3.parameters():
        param.requires_grad = True
        
    model.cuda()

    optimizer = torch.optim.Adam([{'params': model.conv_block4.parameters()},
                                  {'params': model.conv_block5.parameters()},
                                  {'params': model.classifier.parameters()},
                                  {'params': model.score_pool4.parameters()},
                                  {'params': model.score_pool3.parameters()}],
                                  lr=args.l_rate)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, threshold=1e-5)

    loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f Image: %d Percentage: %.2f" % (epoch+1, args.n_epoch, loss.data[0], (i+1) * args.batch_size, (((i+1) * args.batch_size) / 8820) * 100))

        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        scheduler.step(score['Mean IoU : \t'])

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "models/{}_{}_best_model_2.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()
    train(args)
