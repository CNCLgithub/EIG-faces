import argparse
import os
import shutil
import time
import datetime

import numpy as np
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from utils import config

from models.eig.networks.network import EIG
#from models.eig.networks.network_two_fc import EIG_two_fc
#from models.eig.networks.network_no_fc import EIG_no_fc
import datasets

#from neural_analysis.test_import import Neural_Tester

CONFIG = config.Config()

def main():
    parser = argparse.ArgumentParser(description='Training EIG')
    parser.add_argument('--out', type = str, default = '', metavar='PATH',
                        help='Directory to output the result if other than what is specified in the config')
    parser.add_argument("--image", "-is", type = int, nargs="+",
                        default = (3,227,227), help = "Image size. Def: (3, 227, 227)")
    parser.add_argument("--z-size", "-zs", type = int, metavar="N",
                        default = 404, help = "Size of z layer. Default: 404")
    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 20)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    global args
    args = parser.parse_args()
    # create model using the pretrained alexnet.
    print("=> Construct the model...")
    
    model = EIG(args.z_size)
    print(model)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss(size_average=True).cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                args.lr)

    if args.out == '':
        out_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'eig', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_path)
    else:
        out_path = args.out
    if args.resume != '':
        print("=> loading checkpoint '{}'".format(args.resume))
        resume_path = args.resume
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    print("Current run location: {}".format(out_path))

    print('dataset reader begin')
    # Initialize the datasets
    d_full = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling.hdf5'), raw_image = True, input_shape = args.image)
    d_segment = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_segment.hdf5'), raw_image = True, input_shape = args.image)

    val_loader_full = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_val.hdf5'), raw_image = True, input_shape = args.image, augment = False)
    val_loader_segment = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_val_segment.hdf5'), raw_image = True, input_shape = args.image, augment = False)
    print('dataset reader end')

    for epoch in range(args.start_epoch, args.epochs):

        print('training begin')
        # train for one epoch
        train(d_full, d_segment, model, criterion, optimizer, epoch)

        # validate
        avg_loss = validate(val_loader_full, val_loader_segment, model, criterion, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, False, out_path + 'checkpoint_bfm.pth.tar')



def train(d_full, d_segment, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    dtype = torch.FloatTensor

    # switch to train mode
    model.train()

    end = time.time()

    # shuffle trials
    N = len(d_full)
    indices = np.random.permutation(N)
    counter = 0

    # create batches for effecient IO
    batch_size_half = int(args.batch_size / 2)

    batches = np.array_split(indices, N/args.batch_size)

    images = np.zeros((args.batch_size, 3, 227, 227))
    targets = np.zeros((args.batch_size, args.z_size))

    for batch in batches:

        trials = d_full[batch[:batch_size_half]]
        trials.extend(d_segment[batch[batch_size_half:args.batch_size]])

        images, targets = zip(*trials)
        images = np.array(images)
        targets = np.array(targets) * 10

        data_time.update(time.time() - end)

        loss = 0
        input_var = torch.from_numpy(images).type(dtype).cuda()
        target_var = torch.from_numpy(targets).type(dtype).cuda()

        _, _, _, out = model(input_var, segment=False)

        loss = criterion(out, target_var)

        # record loss
        losses.update(loss.data)
        
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
                      epoch, counter * args.batch_size, len(d_full),
                      batch_time=batch_time,
                      data_time=data_time, loss=losses))
            frame_len = 0

        counter += 1


def validate(val_loader_full, val_loader_segment, model, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    dtype = torch.FloatTensor

    # switch to train mode
    model.eval()

    end = time.time()

    # shuffle trials
    N = 5000
    indices = np.random.permutation(N)
    counter = 0

    # create batches for effecient IO
    batch_size_half = int(args.batch_size / 2)

    batches = np.array_split(indices, N/args.batch_size)

    images = np.zeros((args.batch_size, 3, 227, 227))
    targets = np.zeros((args.batch_size, args.z_size))

    for batch in batches:


        trials = val_loader_full[batch[:10]]
        trials.extend(val_loader_segment[batch[10:]])

        images, targets = zip(*trials)
        images = np.array(images)
        targets = np.array(targets) * 10

        data_time.update(time.time() - end)

        loss = 0
        input_var = torch.from_numpy(images).type(dtype).cuda()
        target_var = torch.from_numpy(targets).type(dtype).cuda()

        _, _, _, out = model(input_var, segment=False)

        loss = criterion(out, target_var)

        # record loss
        losses.update(loss.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, counter * args.batch_size, N,
                      batch_time=batch_time,
                      data_time=data_time, loss=losses))
            frame_len = 0

        counter += 1

    return losses.avg


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


if __name__ == '__main__':
    main()

