import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from PIL import Image

from models.id.networks.network_light_depth import VGG_Light_Depth
import datasets

from utils import config
CONFIG = config.Config()

def main():
    parser = argparse.ArgumentParser(description='Training eig')
    parser.add_argument('--out', type = str, default = '', metavar='PATH',
                        help='Directory to output the result if other than what is specified in the config')
    parser.add_argument("--image", "-is", type = int, nargs="+",
                        default = (3,224,224), help = "Image size. Def: (3, 224, 224)")
    parser.add_argument('--light_or_depth', type = str, default='light',
                        help = 'linearly decode lighting of the scene or the depth of the face. Options: light, depth')
    parser.add_argument('--level', type = str,
                        default = 'sfcl', help = 'level at which to decode the network, can be sfcl, ffcl, and tcl')
    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 20)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    global args
    args = parser.parse_args()

    # model
    print("=> Construct the model...")
    model = VGG_Light_Depth(1, args.level)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    args.lr)

    out_path = args.out + args.level + '/'

    if args.out == '':
        out_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'vgg', args.light_or_depth, args.level, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_path)
    else:
        out_path = args.out

    if args.resume != '':
        print("=> loading checkpoint '{}'".format(args.resume))
        resume_path = out_path + '/checkpoint_bfm.pth.tar'
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    print("Current run location: {}".format(out_path))


    # Initialize the datasets
    assert args.light_or_depth in ['light', 'depth'], 'light-or-depth can be either light or depth; e.g., --light-or-depth light'
    if args.light_or_depth == 'light':
        train_loader = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'vgg_light.hdf5'), raw_image = True, input_shape = args.image, augment = False)
    else:
        train_loader = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'vgg_depth.hdf5'), raw_image = True, input_shape = args.image, augment = False)

    for epoch in range(args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, False, out_path + '/checkpoint_bfm.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    dtype = torch.FloatTensor

    # switch to train mode
    #model.train()

    end = time.time()

    # shuffle trials
    N = 10000

    indices = np.random.permutation(N)
    counter = 0

    batches = np.array_split(indices, N/args.batch_size)

    for batch in batches:
        # load batch

        with train_loader.withbackground(False):
            trials = train_loader[batch]
        

        # if you want to run the batch through the network
        # instead of going one trial at a time, you can do:
        images, targets = zip(*trials)
        images = np.array(images)
        input_var = torch.cuda.FloatTensor(images)
        target_var = torch.from_numpy(np.expand_dims(np.array(targets)[:, args.light_or_depth], 1)).type(dtype).cuda()
        data_time.update(time.time() - end)

        loss = 0

        out = model(input_var)

        loss = criterion(out, target_var)

        # record loss
        losses.update(loss.data[0])
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, counter * args.batch_size, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time, loss=losses))
            frame_len = 0

        counter += 1

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

