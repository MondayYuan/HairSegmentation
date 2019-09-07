'''
train hair segmentation in CelebA dataset
'''
import torch
from torch.utils.data import  random_split
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

from utils.dataset import CelebA, Figaro, OurDataset, CelebAMaskHQ, SpringFace, SpringHair
from utils.view_segmentation import overlay_segmentation_mask
from utils.transform import UnNormalize
from utils.loss import FocalLoss

import numpy as np
import time

from tensorboardX import SummaryWriter

def train(args, model):
    writer = SummaryWriter(comment=args.model)

    crop_size = None
    if args.crop_size:
        crop_size = args.crop_size.split('x')
        crop_size = tuple([int(x) for x in crop_size])
    if args.dataset == 'CelebA':
        dataset = CelebA(args.datadir, resize=args.resize, argument=args.argument)
    elif args.dataset == 'Figaro':
        dataset = Figaro(args.datadir, resize=args.resize, crop_size=crop_size, argument=args.argument)
    elif args.dataset == 'Our':
        dataset = OurDataset(args.datadir, resize=args.resize, argument=args.argument)
    elif args.dataset == 'CelebAMaskHQ':
        dataset = CelebAMaskHQ(args.datadir, resize=args.resize, argument=args.argument)
    elif args.dataset == 'SpringFace':
        dataset = SpringFace(args.datadir, resize=args.resize, argument=args.argument)
    elif args.dataset == 'SpringHair':
        dataset = SpringHair(args.datadir, resize=args.resize, crop_size=crop_size, argument=args.argument)
    else:
        print('Fail to find the dataset')
        raise ValueError

    num_train = int(args.train_val_rate * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1))
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if args.gpu else "cpu")

    writer.add_graph(model, torch.zeros(args.batch_size, 3, 218, 178).to(device))

    model.train()

    if args.model == 'unet':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.model == 'denseunet':
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = FocalLoss(gamma=2).to(device)
    else:
        print('Fail to find the net')
        raise ValueError

    optimizer = Adam(model.parameters(), lr=args.lr)

    max_mean_iu = -999

    n_steps = 0
    for i_epoch in range(args.num_epochs):
        model.train()

        for step, (images, labels, _) in enumerate(train_loader):
            # print(step)
            inputs = images.to(device)
            targets = labels.to(device)
            # print('input', inputs.shape)
            # print('target', targets.shape)
            outputs = model(inputs).squeeze(1)
            # print('output', outputs.shape)

            optimizer.zero_grad()

            loss = criterion(outputs, targets)
            loss.backward()

            writer.add_scalar('/train/loss', loss, n_steps)

            optimizer.step()
            n_steps += 1

            if n_steps % args.iters_eval == 0 or \
                    (i_epoch == args.num_epochs - 1 and step == len(train_loader) - 1):
                result = evaluate_segment(args, model, val_dataset)
                time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'epoch = {i_epoch}, iter = {n_steps},  train_loss = {loss} ---{time_stamp}')
                print(result)

                for key in result.keys():
                    writer.add_scalar(f'/val/{key}', result[key], i_epoch)

                for i in range(6):
                    img, label, img_name = val_dataset[i]
                    writer.add_text('val_img_name', img_name, i_epoch)

                    output = model(img.unsqueeze(0).to(device))
                    mask = torch.argmax(output, 1)

                    img = UnNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])(img)
                    img_masked = overlay_segmentation_mask(img, mask, inmode='tensor', outmode='tensor')

                    mask = mask * 127
                    label = label * 127

                    writer.add_image(f'/val/image/{img_name}', img, n_steps)
                    writer.add_image(f'/val/label/{img_name}', label.unsqueeze(0), n_steps)
                    writer.add_image(f'/val/image_masked/{img_name}', img_masked, n_steps)
                    writer.add_image(f'/val/mask/{img_name}', mask, n_steps)

                if result['mean_iu'] > max_mean_iu:
                    max_mean_iu = result['mean_iu']
                    torch.save(model.state_dict(), args.save_path)

def evaluate_segment(args, model, val_set):
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

    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(labels_truth, labels_predict, 2)

    return {
        'acc': acc,
        'acc_cls': acc_cls,
        'mean_iu': mean_iu,
        'fwavacc': fwavacc
    }
