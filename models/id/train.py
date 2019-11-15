import argparse
import os
import random
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from PIL import Image

import datasets
from models.id.networks.network import VGG

from utils import config
CONFIG = config.Config()

def main():
    parser = argparse.ArgumentParser(description='Training eig')

    parser.add_argument('--out', type = str, default = '', metavar='PATH',
                        help='Directory to output the result if other than what is specified in the config')
    parser.add_argument('--imageset', default='bfm', type=str,
                        help='Train with BFM (bfm) images or FIV (fiv) images?')
    parser.add_argument("--image", "-is", type = int, nargs="+",
                        default = (3,224,224), help = "Image size. Def: (3, 224, 224)")
    parser.add_argument("--num-classes", "-nc", type = int, metavar="N",
                        default = 25, help = "Number of unique individual identities. Default: 25.")
    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
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

    assert args.imageset in ['bfm', 'fiv'], 'set imageset to either bfm or fiv; e.g., --imageset fiv'

    # model
    print("=> Construct the model...")
    if args.imageset == 'fiv':
        model = VGG(25)
    else:
        model = VGG(500)
    model.cuda()
    model.neural_test = False

    print(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
                                       model.parameters()), args.lr)

    if args.out == '':
        out_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'vgg', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_path)
    else:
        out_path = args.out
    if args.resume != '':
        print("=> loading checkpoint '{}'".format(args.resume))
        resume_path = args.resume
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    print("Current run location: {}".format(out_path))

    if args.imageset == 'bfm':
        train_loader = datasets.BFMId(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_ID.hdf5'), raw_image = True, input_shape = args.image)
    else:
        train_loader = datasets.BFMId(os.path.join(CONFIG['PATHS', 'databases'], 'FV_segment_patchup.hdf5'), raw_image = True, input_shape = args.image)

    val_loader = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_id_ft.hdf5'), raw_image = True, input_shape = args.image, augment = False)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        if args.imageset == 'bfm':
            # validate
            avg_loss = validate(val_loader, model, criterion, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, False, os.path.join(out_path, 'checkpoint_' + args.imageset + '.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    dtype = torch.FloatTensor

    # switch to train mode
    model.train()

    end = time.time()

    # shuffle trials
    N = len(train_loader)
    if args.imageset == 'fiv':
        N = 10000

    indices = np.random.permutation(N)
    counter = 0

    batches = np.array_split(indices, N/args.batch_size)

    for batch in batches:
        # load batch
        trials = train_loader[batch]
        
        # if you want to run the batch through the network
        # instead of going one trial at a time, you can do:
        images, targets = zip(*trials)
        images = np.array(images)

        input_var = torch.cuda.FloatTensor(images)

        targets = np.floor(batch/400).astype(int)
        target_var = torch.cuda.LongTensor(targets)
        data_time.update(time.time() - end)

        loss = 0

        out = model(input_var)

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
                      epoch, counter * args.batch_size, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time, loss=losses))
            frame_len = 0

        counter += 1



def validate(val_loader, model, criterion, epoch):
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

    batches = np.array_split(indices, N/args.batch_size)

    for batch in batches:
        # load batch
        trials = val_loader[batch]

        # if you want to run the batch through the network
        # instead of going one trial at a time, you can do:
        images, targets = zip(*trials)
        images = np.array(images)
        input_var = torch.cuda.FloatTensor(images)

        targets = np.floor(batch/400).astype(int)
        target_var = torch.cuda.LongTensor(targets)
        data_time.update(time.time() - end)

        loss = 0

        out = model(input_var)

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

