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

from models.eig.networks.network_classifier import EIG_classifier
import datasets

def main():
    parser = argparse.ArgumentParser(description='Fine the EIG networks f2 and train f3 to obtain EIG_CLASSIFIER')
    parser.add_argument('--out', type = str, default = '', metavar='PATH',
                        help='Directory to output the result if other than what is specified in the config')
    parser.add_argument('--imageset', default='bfm', type=str,
                        help='Train with BFM (bfm) images or FIV (fiv) images?')
    parser.add_argument("--image", "-is", type = int, nargs="+",
                        default = (3,227,227), help = "Image size. Def: (3, 227, 227)")
    parser.add_argument("--num-classes", "-nc", type = int, metavar="N",
                        default = 25, help = "Number of unique individual identities. Default: 25.")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=20, type=int,
                        metavar='N', help='mini-batch size (default: 20)')
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    global args
    args = parser.parse_args()

    assert args.imageset in ['bfm', 'fiv'], 'set imageset to either bfm or fiv; e.g., --imageset fiv'

    # create model using the pretrained alexnet.
    print("=> Construct the model...")
    
    model = EIG_classifier()
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # finetune SFCL and the new identity layer
    optimizer = torch.optim.SGD([
        {'params': model.fc_layers.parameters(), 'lr':  0.0005},
        {'params': model.classifier.parameters(), 'lr': 0.0005}
    ])

    if args.out == '':
        out_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'eig_classifier', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(out_path)
    else:
        out_path = args.out
    print("=> loading checkpoint '{}'".format(args.resume))
    resume_path = args.resume
    checkpoint = torch.load(resume_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

    print("Current run location: {}".format(out_path))
    
    # Initialize the datasets
    if args.imageset == 'fiv':
        train_loader = datasets.BFMId(os.path.join(CONFIG['PATHS', 'databases'], 'FIV_segment_bootstrap.hdf5'), raw_image = True, input_shape = args.image)
        epochs = args.start_epoch + 20
     else:
        d_full = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], 'bfm09_backface_culling_id_ft.hdf5'), raw_image = True, input_shape = args.image)
        d_segment = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databses'], 'bfm09_backface_culling_id_ft_segment.hdf5'), raw_image = True, input_shape = args.image)
        train_loader = (d_full, d_segment)
        epochs = args.start_epoch + 2


    for epoch in range(args.start_epoch, epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
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
    N = 10000
    indices = np.random.permutation(N)
    counter = 0

    batches = np.array_split(indices, N/args.batch_size)

    if args.imageset == 'bfm':
        train_loader_full, train_loader_segment = train_loader
        batch_size_half = args.batch_size / 2

    for batch in batches:

        if args.imageset == 'fiv':
            trials = train_loader[batch]
        else:
            trials = train_loader_full[batch[:batch_size_half]]
            trials.extend(train_loader_segment[batch[batch_size_half:]])

        images, targets = zip(*trials)
        images = np.array(images)
        targets = np.floor(batch/400).astype(int)
        data_time.update(time.time() - end)

        loss = 0
        input_var = torch.from_numpy(images).type(dtype).cuda()
        target_var = torch.cuda.LongTensor(targets)

        out = model(input_var, segment=False)

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
                      epoch, counter * args.batch_size, N,
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

