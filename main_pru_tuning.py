
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from tqdm import *

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_taskonomy
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mt

from engine_mt import train_one_epoch, evaluate
from util.AutomaticWeightedLoss import AutomaticWeightedLoss

from fvcore.nn import FlopCountAnalysis, flop_count_str
from ptflops import get_model_complexity_info



# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 44875 main_pru_tuning.py \
#         --batch_size 20 \
#         --epochs 100 \
#         --input_size 224 \
#         --blr 5e-4 --weight_decay 0.05 \
#         --warmup_epochs 10 \
#         --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#         --model mtvit_taskgate_small_att_mlp \
#         --drop_path 0.1 \
#         --times 1 \
#         --cycle \
#         --the_task class_scene \
#         --copy fixnew_mtvit_taskgate_small_att_mlp_7 \
#         --exp-name pruning_debug \




def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/gpfs/u/home/AICD/AICDzich/scratch/vl_eval_data/ILSVRC2012', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/MTMoe',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/logs_MTMoe',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument("--exp-name", type=str, required=True, help="Name for experiment run (used for logging)")

    parser.add_argument('--times', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--tasks', default=2, type=int,
                        help='number of tasks')

    parser.add_argument('--ori_tasks', default=14, type=int,
                        help='number of original tasks')

    parser.add_argument('--eval_all', action='store_true')
    parser.add_argument('--cycle', action='store_true')
    parser.add_argument('--only_gate', action='store_true')
    parser.add_argument('--dynamic_lr', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--the_task', type=str, default='class_object',
                        help='The only one task')

    parser.set_defaults(only_gate=False)
    parser.set_defaults(cycle=False)
    parser.set_defaults(eval_all=False)
    parser.set_defaults(dynamic_lr=False)
    parser.set_defaults(visualize=False)

    parser.add_argument('--visualizeimg', action='store_true')
    parser.set_defaults(visualizeimg=False)

    parser.add_argument('--copy', default='',
                        help='copy from exp for pruning')
    parser.add_argument('--vis_file', default='',
                        help='visualize file')
    parser.add_argument('--thresh', type=float, default=3, 
                        help='threshold for copying the expert')

    return parser





def main(args):
    if args.tasks == 2:
        args.img_types = [args.the_task, 'rgb']
    else:
        assert False

    if args.ori_tasks == 15:
        args.ori_img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
    elif args.ori_tasks == 14:  # no semantic_seg
        args.ori_img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
    elif args.ori_tasks == 10:  # no semantic_seg
        args.ori_img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
    elif args.ori_tasks == 9:  # no semantic_seg
        args.ori_img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
    elif args.ori_tasks == 7:  # no semantic_seg
        args.ori_img_types = ['class_object', 'depth_euclidean', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'edge_occlusion']
    elif args.ori_tasks == 2:
        args.ori_img_types = [args.the_task, 'rgb']
    else:
        assert False

    # make dir
    args.output_dir = os.path.join(args.output_dir, str(args.exp_name))
    args.log_dir = os.path.join(args.log_dir, str(args.exp_name))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # check if there is already 
    files = os.listdir(args.output_dir)
    for file in files:
        if file[:10] == 'checkpoint': #
            print('resume', os.path.join(args.output_dir, file))
            args.resume = os.path.join(args.output_dir, file)
        
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_taskonomy(is_train=True, args=args)
    dataset_val = build_taskonomy(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        args.dist_eval = True
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    
    model = models_mt.__dict__[args.model](
        args.img_types,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    checkpoint_model = model.delete_ckpt(args)

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.frozen()

    model.patch_embed.proj.requires_grad = False
    model.pos_embed.requires_grad = False
    model.cls_token.requires_grad = False
    for blk in model.blocks:
        blk.attn.kv_proj.requires_grad = False
        blk.attn.q_proj.experts.requires_grad = False
        blk.attn.q_proj.output_experts.requires_grad = False

        blk.mlp.experts.requires_grad = False
        blk.mlp.output_experts.requires_grad = False

    model.to(device)
    # print('pos_emb: ', model.pos_embed.requires_grad)
    # print('patch: ', model.patch_embed.proj.requires_grad)

    # model.requires_grad = False
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name,  p.requires_grad)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))
        print('model_name:', args.model)
        
        if model_without_ddp.moe_type == 'FLOP' or model_without_ddp.ismoe == False:
            t_mg_types = [type_ for type_ in args.img_types if type_ != 'rgb']
            flops = FlopCountAnalysis(model, (torch.randn(1,3,224,224).to(device), t_mg_types[0], True))
            print('Model total flops: ', flops.total()/1000000000, 'G ', t_mg_types[0])


        print('number of params (M): %.2f' % (n_parameters / 1.e6))
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        print('len train: ', len(dataset_train))
        print('len val: ', len(dataset_val))

    args.distributed = True
    if args.distributed:
        if args.tasks == 2:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    AWL = AutomaticWeightedLoss(args.tasks-1)
    AWL.to(device)

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
        AWL=AWL
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
        
    for epoch in range(args.start_epoch, args.epochs * args.times):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, None,
            AWL=AWL,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device, AWL, args)

        if log_writer is not None:
            for _key, value in test_stats.items():
                log_writer.add_scalar('perf/test_' + str(_key), value, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
        pass
    else:
        args.output_dir = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/MTMoe'
        args.log_dir = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/logs_MTMoe'
    # args.output_dir = '/data/zitianchen/work_dirs/MTMoe'
    # args.log_dir = '/data/zitianchen/work_dirs/logs_MTMoe'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
