from argparse import  ArgumentParser
from nets.unet import UNet
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

    assert model is not None, f'model {args.model} not available'

    if args.gpu:
        model = model.cuda()

    if args.double_gpus:
        model = nn.DataParallel(model, [0, 1])

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
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--double-gpus', action='store_true', default=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--load-path')
    parser.add_argument('--n-classes', type=int, default=2)

    subparser = parser.add_subparsers(dest='mode')
    subparser.required = True

    parser_train = subparser.add_parser('train')
    parser_train.add_argument('--dataset', required=True)
    parser_train.add_argument('--datadir', default='data')
    parser_train.add_argument('--num-epochs', type=int, default=100)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--lr', type=float, default=0.1)
    parser_train.add_argument('--num-workers', type=int, default=2)
    parser_train.add_argument('--epochs-eval', type=int, default=5)
    parser_train.add_argument('--save-dir', default='./model')
    parser_train.add_argument('--argument', action='store_true', default=False)

    parser_test = subparser.add_parser('test')
    parser_test.add_argument('--image')
    parser_test.add_argument('--video')
    parser_test.add_argument('--save')
    parser_test.add_argument('--resize', action='store_true', default=False)
    parser_test.add_argument('--remove-small-area', action='store_true', default=False)
    parser_test.add_argument('--detector')

    main(parser.parse_args())
