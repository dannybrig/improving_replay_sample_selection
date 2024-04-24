# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
from torch.nn import functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer, icarl_replay


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser

def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()
    samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_l = self.buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
            mem_buffer.add_data(
                examples=_y_x[:samples_per_class],
                labels=_y_y[:samples_per_class],
                logits=_y_l[:samples_per_class]
            )

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x
    classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for x, y, not_norm_x in loader:
        mask = (y >= classes_start) & (y < classes_end)
        x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
        if not x.size(0):
            continue
        x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
        a_x.append(not_norm_x.to('cpu'))
        a_y.append(y.to('cpu'))
        feats = self.net(norm_trans(not_norm_x), returnt='features')
        outs = self.net.classifier(feats)
        a_f.append(feats.cpu())
        a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means
    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        feats = a_f[idx]
        mean_feat = feats.mean(0, keepdim=True)

        running_sum = torch.zeros_like(mean_feat)
        i = 0
        while i < samples_per_class and i < feats.shape[0]:
            cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

            idx_min = cost.argmin().item()

            mem_buffer.add_data(
                examples=_x[idx_min:idx_min + 1].to(self.device),
                labels=_y[idx_min:idx_min + 1].to(self.device),
                logits=_l[idx_min:idx_min + 1].to(self.device)
            )

            running_sum += feats[idx_min:idx_min + 1]
            feats[idx_min] = feats[idx_min] + 1e6
            i += 1

    assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

    self.net.train(mode)

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        
        self.buffer = Buffer(self.args.buffer_size, self.device)
        
        self.class_means = None
        self.old_net = None
        self.task = 0
        
    def observe(self, inputs, labels, not_aug_inputs):
        
        # iCaRL Stuff
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())
            
        self.class_means = None
        if self.task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
            
        # DER Stuff
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()
        #self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()
    
    def begin_task(self, dataset):
        icarl_replay(self, dataset)
    
    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            fill_buffer(self, self.buffer, dataset, self.task)
        self.task += 1
        self.class_means = None