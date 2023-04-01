import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_cycle_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    the_epoch = epoch % args.epochs

    if args.dynamic_lr:  # Limit the max lr
        t_d = 0.5 ** (epoch//args.epochs)
    else:
        t_d = 1.0
    
    if the_epoch < args.warmup_epochs:
        lr = (args.lr * t_d) * the_epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + ((args.lr * t_d) - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (the_epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
