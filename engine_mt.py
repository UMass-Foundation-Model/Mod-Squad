# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, AWL=None,
                    args=None):
    model.train(True)
    AWL.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (data) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        
        if data_iter_step % accum_iter == 0:
            if args.cycle:
                lr_sched.adjust_cycle_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            else:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)
        samples = data['rgb'].to(device, non_blocking=True)
        z_loss = 0
        loss = 0
        the_loss = {}
        loss_list = []
        tot_loss = 0

        # with torch.autograd.set_detect_anomaly(True):
        with torch.cuda.amp.autocast():
            for task in args.img_types:
                if 'rgb' in task:
                    continue
                outputs, aux_loss = model(samples, task)
                z_loss = z_loss + aux_loss

                targets = data[task].to(device, non_blocking=True)
                if 'class' in task:
                    # task_loss = criterion(outputs, targets.view(-1))
                    task_loss = F.mse_loss(outputs, targets.squeeze(1))
                elif 'segment_semantic' in task:
                    task_loss = criterion(outputs, targets)
                elif 'normal' in task:
                    targets = targets.permute(0,2,3,1)
                    task_loss = (1 - (outputs*targets).sum(-1) / torch.norm(outputs, p=2, dim=-1) / torch.norm(targets, p=2, dim=-1)).mean()
                else:
                    if outputs.shape[-1] == 1:
                        outputs = outputs.view(outputs.shape[:-1])
                    elif outputs.shape[-1] == 3:
                        outputs = outputs.permute(0,3,1,2)
                    task_loss = F.mse_loss(outputs, targets)
                # loss = loss + task_loss
                tot_loss = tot_loss + task_loss.item()
                the_loss[task] = task_loss
                loss_list.append(the_loss[task])

        loss = AWL(loss_list)
        loss_value = loss.item()

        loss = loss + z_loss
        if torch.is_tensor(z_loss):
            z_loss_value = z_loss.item()
        else:
            z_loss_value = z_loss

        # tot_loss_value = tot_loss.item()

        the_loss_value = {}
        for _key, value in the_loss.items():
            the_loss_value[_key] = value.item()

        loss = torch.clamp(loss, min=-1000, max=1000)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            loss = torch.clamp(loss, min=-1000, max=1000)
            sys.exit(1)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        for _key, value in the_loss_value.items():
            metric_logger.meters[_key].update(value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        tot_loss_reduce = misc.all_reduce_mean(tot_loss)
        if torch.is_tensor(z_loss):
            z_loss_value_reduce = misc.all_reduce_mean(z_loss_value)
        else:
            z_loss_value_reduce = 0

        the_loss_value_reduce = {}
        for _key, value in the_loss_value.items():
            the_loss_value_reduce[_key] = misc.all_reduce_mean(value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('z_loss', z_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('tot_loss', tot_loss_reduce, epoch_1000x)
            
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            for _key, value in the_loss_value_reduce.items():
                log_writer.add_scalar('multitask/' + _key, value, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('params: ', AWL.params)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, AWL, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    AWL.eval()

    for data in metric_logger.log_every(data_loader, 10, header):
        images = data['rgb']
        # target = batch[-1]
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        # # compute output
        # with torch.cuda.amp.autocast():
        #     output, _ = model(images)
        #     loss = criterion(output, target)

        the_loss = {}
        loss_list = []
        tot_loss = 0
        with torch.cuda.amp.autocast():
            for task in args.img_types:
                if 'rgb' in task:
                    continue
                outputs, _ = model(images, task)

                targets = data[task].to(device, non_blocking=True)
                if 'class' in task:
                    # task_loss = criterion(outputs, targets.view(-1))
                    task_loss = F.mse_loss(outputs, targets.squeeze(1))
                elif 'segment_semantic' in task:
                    task_loss = criterion(outputs, targets)
                elif 'normal' in task:
                    targets = targets.permute(0,2,3,1)
                    task_loss = (1 - (outputs*targets).sum(-1) / torch.norm(outputs, p=2, dim=-1) / torch.norm(targets, p=2, dim=-1)).mean()
                else:
                    if outputs.shape[-1] == 1:
                        outputs = outputs.view(outputs.shape[:-1])
                    elif outputs.shape[-1] == 3:
                        outputs = outputs.permute(0,3,1,2)
                    task_loss = F.mse_loss(outputs, targets)

                the_loss[task] = task_loss
                loss_list.append(the_loss[task])
                tot_loss = tot_loss + task_loss.item()

        loss = AWL(loss_list)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(tot_loss=tot_loss)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        for _key, value in the_loss.items():
            metric_logger.meters[_key].update(value.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('test Result: ', ' '.join(str(a) + ':' + str(b.global_avg) for (a,b) in metric_logger.meters.items()))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
