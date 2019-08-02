'''
train hair segmentation in CelebA dataset
'''
import torch
from torch.utils.data import  random_split
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

from utils.dataset import CelebA, Figaro
from utils.view_segmentation import overlay_segmentation_mask
from utils.transform import UnNormalize
from utils.dataloader import DataLoaderX

import numpy as np
import time
import os
import cv2

from PIL import Image
from torchvision.transforms import ToPILImage

from tensorboardX import SummaryWriter

def train(args, model):
    writer = SummaryWriter(comment=args.model)

    if args.dataset == 'CelebA':
        train_dataset = CelebA(args.datadir, mode='train', n_classes=args.n_classes, argument=args.argument)
        val_dataset = CelebA(args.datadir, mode='val', n_classes=args.n_classes, argument=args.argument)
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'Figaro':
        train_dataset = Figaro(args.datadir, mode='train', argument=args.argument)
        val_dataset = Figaro(args.datadir, mode='val', argument=args.argument)
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    elif args.dataset == 'CelebA+Figaro':
        train_dataset_celeba = CelebA('data/CelebA', mode='train', n_classes=args.n_classes, argument=args.argument)
        train_dataset_figaro = Figaro('data/Figaro', mode='trainval', argument=args.argument)
        val_dataset = CelebA('data/CelebA', mode='val', n_classes=args.n_classes, argument=args.argument)
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
        train_loader = DataLoaderX(train_dataset_celeba, train_dataset_figaro, args.batch_size, worker_init_fn)
    else:
        print('--dataset should be CeleA or Figaro or CelebA+Figaro')
        raise TypeError


    device = torch.device("cuda" if args.gpu else "cpu")

    writer.add_graph(model, torch.zeros(args.batch_size, 3, 218, 178).to(device))

    model.train()

    criterion = nn.CrossEntropyLoss().to(device)

    # if args.model == 'fastDeepMatting':
    #     optimizer = Adam(model.parameters(), 5)
    if args.model == 'unet':
        optimizer = Adam(model.parameters(), lr=args.lr)

    max_mean_iu = -999

    for i_epoch in range(args.num_epochs):
        model.train()

        for step, (images, labels, _) in enumerate(train_loader):
            # print(step)
            inputs = images.to(device)
            targets = labels.to(device)
            # print('input', inputs.shape)
            # print('target', targets.shape)
            outputs = model(inputs)
            # print('output', outputs.shape)

            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()

            writer.add_scalar('/train/loss', loss.item(), i_epoch)

            optimizer.step()

        if i_epoch % args.epochs_eval == 0:
            acc, acc_cls, mean_iu, fwavacc = evaluate(args, model, val_dataset)
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('epoch ={}, loss = {}, macc = {}, mean_iu = {} ---{}'.format(i_epoch, loss.item(), acc_cls, mean_iu, time_stamp))

            writer.add_scalar('/val/acc', acc, i_epoch)
            writer.add_scalar('/val/acc_cls', acc_cls, i_epoch)
            writer.add_scalar('/val/mean_iu', mean_iu, i_epoch)
            writer.add_scalar('/val/fwavacc', fwavacc, i_epoch)

            img, label, img_name = val_dataset[2]
            # img, label, img_name = train_dataset[0]

            writer.add_text('val_img_name', img_name, i_epoch)

            output = model(img.unsqueeze(0).to(device))
            mask = torch.argmax(output, 1)

            img = UnNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])(img)
            img_masked = overlay_segmentation_mask(img, mask, inmode='tensor', outmode='tensor')

            mask = mask * 127 + 1
            label = label * 127 + 1

            writer.add_image('/val/image', img, i_epoch)
            writer.add_image('/val/label', label.unsqueeze(0), i_epoch)
            writer.add_image('/val/image_masked', img_masked, i_epoch)
            writer.add_image('/val/mask', mask, i_epoch)

            if mean_iu > max_mean_iu:
                max_mean_iu = mean_iu
                filename = os.path.join(args.save_dir, f'{args.model}.pth')
                torch.save(model.state_dict(), filename)


def evaluate(args, model, val_set):
    model.eval()
    device = torch.device("cuda" if args.gpu else "cpu")
    val_loader = DataLoader(val_set,
        num_workers=1, batch_size=args.batch_size, shuffle=False)
    labels_truth = []
    labels_predict = []
    for step, (images, labels, _) in enumerate(val_loader):
        inputs = images.to(device)
        targets = labels.to(device)

        outputs = model(inputs).max(1)[1]

        labels_truth.append(targets.cpu().numpy())
        labels_predict.append(outputs.cpu().numpy())

    def label_accuracy_score(label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        """
        hist = np.zeros((n_class, n_class))

        def _fast_hist(label_true, label_pred, n_class):
            mask = (label_true >= 0) & (label_true < n_class)
            hist = np.bincount(
                n_class * label_true[mask].astype(int) +
                label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
            return hist

        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(labels_truth, labels_predict, args.n_classes)

    return acc, acc_cls, mean_iu, fwavacc
