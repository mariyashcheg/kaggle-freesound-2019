import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

import data
import models
import train
import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='/data/runs/')
    parser.add_argument('--epochs', default=6)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--model', default='simple_cgrnn')
    parser.add_argument('--lr', default='const')
    return parser.parse_args()


def cyclic_learning_rate(stepsize, min_lr=1e-4, max_lr=3e-4, scaler_mode='clr', gamma=0.95):
    if scaler_mode == 'clr':
        scaler = lambda x: 1
    elif scaler_mode == 'exp_clr':
        scaler = lambda x: gamma ** x

    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    def relative(it, stepsize):
        cycle = np.floor(1 + it / float(2 * stepsize))
        x = abs(it / float(stepsize) - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def main(args):
    np.random.seed(432)
    torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))
    val_writer = SummaryWriter(os.path.join(experiment_path, 'val_logs'))
    trainer = train.Trainer(train_writer, val_writer)

    # todo: add config
    train_transform = data.build_preprocessing(args.model)
    eval_transform = data.build_preprocessing(args.model)

    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform
    evalds.transform = eval_transform

    trainloader = DataLoader(trainds, batch_size=int(args.batch_size), shuffle=True, num_workers=8, pin_memory=True)
    evalloader = DataLoader(evalds, batch_size=int(args.batch_size), shuffle=False, num_workers=16, pin_memory=True)

    if args.model == 'simple_cgrnn':
        model = models.cgrnn_simple()
    else:
        model = models.cgrnn(trainds[0]['logmel'].size(1))

    opt = torch.optim.Adam(model.parameters())
    if args.lr == 'const':
        if args.model == 'simple_cgrnn':
            lr_scheduler = lambda x: 1e-4
        else:
            lr_scheduler = lambda x: 4e-4
    else:
        if args.model == 'simple_cgrnn':
            min_lr, max_lr = 3e-5, 4e-4
        else:
            min_lr, max_lr = 2e-4, 6e-4
        lr_scheduler = cyclic_learning_rate(len(trainloader)*4, min_lr=min_lr, max_lr=max_lr, scaler_mode=args.lr, gamma=0.9)


    for epoch in range(int(args.epochs)):
        trainer.train_epoch(model, opt, lr_scheduler, trainloader)
        metrics = trainer.eval_epoch(model, evalloader)

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(),
            loss=metrics['loss'],
            lwlrap=metrics['lwlrap'],
            global_step=trainer.global_step,
        )
        print('epoch: %d    loss: %.4f    lwlrap: %.4f' % (epoch, metrics['loss'], metrics['lwlrap']))
        export_path = os.path.join(experiment_path, 'last.pth')
        torch.save(state, export_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
