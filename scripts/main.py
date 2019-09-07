from argparse import  ArgumentParser
from nets.unet import UNet
from nets.denseunet import DenseUNet
from nets.fastmatting import FastMatting
import numpy as np

import torch
from torch import nn

from train import train
from test import test_image, test_video

SEED = 0

if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def main(args):
    model = None
    if args.model == 'unet':
        model = UNet(args.n_classes)
    elif args.model == 'denseunet':
        model = DenseUNet(2)

    assert model is not None, f'model {args.model} not available'

    if args.gpu is not None:
        model = model.cuda()
        if len(args.gpu) > 0:
            model = nn.DataParallel(model, device_ids=args.gpu)

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))

    if args.mode == 'train':
        train(args, model)
    if args.mode == 'test':
        if args.image:
            test_image(args, model)
        elif args.video:
            test_video(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='*')
    parser.add_argument('--model', required=True)
    parser.add_argument('--load-path')

    subparser = parser.add_subparsers(dest='mode')
    subparser.required = True

    parser_train = subparser.add_parser('train')
    parser_train.add_argument('--dataset', required=True)
    parser_train.add_argument('--datadir', default='data')
    parser_train.add_argument('--resize', type=int, default=-1)
    parser_train.add_argument('--crop-size') #heightxwidth
    parser_train.add_argument('--train-val-rate', type=float, default=0.8)
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--batch-size', type=int, default=32)
    parser_train.add_argument('--lr', type=float, default=0.01)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--iters-eval', type=int, default=10000)
    parser_train.add_argument('--save-path', default='./model/model.pth')
    parser_train.add_argument('--argument', action='store_true')

    parser_test = subparser.add_parser('test')
    parser_test.add_argument('--image')
    parser_test.add_argument('--video')
    parser_test.add_argument('--save')
    parser_test.add_argument('--resize', type=int, default=-1)
    parser_test.add_argument('--remove-small-area', action='store_true', default=False)
    parser_test.add_argument('--detector')
    parser_test.add_argument('--unshow', action='store_true')

    args = parser.parse_args()
    main(args)
