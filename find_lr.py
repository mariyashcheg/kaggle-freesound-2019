import argparse
import numpy as np
import pandas as pd
import os
from tensorboardX import SummaryWriter
from sklearn.metrics import label_ranking_average_precision_score
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
import models
import utils

plt.switch_backend('agg')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/kaggle-freesound-2019')
    parser.add_argument('--outpath', default='/data/runs/')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--model', default='simple_cgrnn')
    return parser.parse_args()

def find_lr(model, train_writer, optimizer, dataloader, lr_begin, lr_end, experiment_path, log_prefix=""):
    global_step = 0
    device = torch.device('cuda')
    model = model.to(device)
    model.train()
    metrics = defaultdict(list)

    for batch in tqdm(dataloader):
        x = batch['logmel'].to(device)
        y = batch['labels'].to(device)

        lr = np.exp((1 - global_step/float(len(dataloader))) * np.log(lr_begin) + global_step/float(len(dataloader)) * np.log(lr_end))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        out = model(x, model.init_hidden(x.size(0)).to(device))
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        probs = out.cpu().data.numpy()
        lrap = label_ranking_average_precision_score(batch['labels'], probs)

        log_entry = dict(
            lrap=lrap,
            loss=loss.item(),
            lr=lr,
        )
        metrics['loss'].append(loss.item())
        metrics['lrap'].append(lrap)
        metrics['lr'].append(lr)
        # metrics['grad_norm'].append(grad_norm(model))

        for name, value in log_entry.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            train_writer.add_scalar(name, value, global_step=global_step)
        global_step += 1

    fig = plt.figure(figsize=(12,9))
    plt.plot(metrics['lr'], metrics['loss'])
    plt.xscale('log')
    plt.grid()
    train_writer.add_figure('loss_vs_lr', fig, global_step=global_step)
    pd.DataFrame({'lr': metrics['lr'], 'loss': metrics['loss']}, index=np.arange(len(metrics['loss']))).to_csv(experiment_path+'/lvsl.csv')

def main(args):
    np.random.seed(432)
    torch.random.manual_seed(432)
    try:
        os.makedirs(args.outpath)
    except OSError:
        pass
    experiment_path = utils.get_new_model_path(args.outpath)
    train_writer = SummaryWriter(os.path.join(experiment_path, 'train_logs'))

    # todo: add config
    train_transform = data.build_preprocessing(args.model)
    trainds, evalds = data.build_dataset(args.datadir, None)
    trainds.transform = train_transform

    trainloader = DataLoader(trainds, batch_size=int(args.batch_size), shuffle=True,
                             num_workers=8, pin_memory=True)

    if args.model == 'simple_cgrnn':
        model = models.cgrnn_simple()
    else:
        model = models.cgrnn(trainds[0]['logmel'].size(1))

    opt = torch.optim.Adam(model.parameters())
    find_lr(model, train_writer, opt, trainloader, lr_begin=1e-5, lr_end=1e-3, experiment_path=experiment_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
