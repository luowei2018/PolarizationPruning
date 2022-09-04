from __future__ import print_function

import argparse
import typing

import numpy as np
import os
import random
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

import common
import models
from common import LossType, compute_conv_flops
from models.common import SparseGate, Identity
from models.resnet_expand import BasicBlock
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training with Polarization')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='training dataset (default: cifar10)')
parser.add_argument("--loss-type", "-loss", dest="loss",
                    choices=list(LossType.loss_name().keys()), help="the type of loss")
parser.add_argument('--lbd', type=float, default=0.0001,
                    help='scale sparse rate (i.e. lambda in eq.2) (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=1.,
                    help='coefficient of mean term in polarization regularizer. deprecated (default: 1)')
parser.add_argument('--t', type=float, default=1.,
                    help='coefficient of L1 term in polarization regularizer (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--max-epoch', type=int, default=None, metavar='N',
                    help='the max number of epoch, default None')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--decay-epoch', type=float, nargs='*', default=[0.5, 0.75],
                    help="the epoch to decay the learning rate (default 0.5, 0.75)")
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, metavar='S', default=None,
                    help='random seed (default: a random int)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, metavar='PATH', required=True,
                    help='path to save prune model')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--log', type=str, metavar='PATH', required=True,
                    help='path to tensorboard log ')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on decay-epoch, number of gammas should be equal to decay-epoch')
parser.add_argument('--bn-init-value', default=0.5, type=float,
                    help='initial value of bn weight (default: 0.5, following NetworkSlimming)')
parser.add_argument('--retrain', type=str, default=None, metavar="PATH",
                    help="Pruned checkpoint for RETRAIN model.")
parser.add_argument('--clamp', default=1.0, type=float,
                    help='Upper bound of the bn scaling factors (only available at Polarization!) (default: 1.0)')
parser.add_argument('--gate', action='store_true', default=False,
                    help='Add an extra scaling factor after the BatchNrom layers.')
parser.add_argument('--backup-path', default=None, type=str, metavar='PATH',
                    help='path to tensorboard log')
parser.add_argument('--backup-freq', default=10, type=float,
                    help='Backup checkpoint frequency')
parser.add_argument('--fix-gate', action='store_true',
                    help='Do not update parameters of SparseGate while training.')
parser.add_argument('--flops-weighted', action='store_true',
                    help='The polarization parameters will be weighted by FLOPs.')
parser.add_argument('--weight-max', type=float, default=None,
                    help='Maximum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--weight-min', type=float, default=None,
                    help='Minimum FLOPs weight. Only available when --flops-weighted is enabled.')
parser.add_argument('--bn-wd', action='store_true',
                    help='Apply weight decay on BatchNorm layers')
parser.add_argument('--target-flops', type=float, default=None,
                    help='Stop when pruned model archive the target FLOPs')
parser.add_argument('--max-backup', type=int, default=None,
                    help='The max number of backup files')
parser.add_argument('--input-mask', action='store_true',
                    help='If use input mask in ResNet models.')
parser.add_argument('--width-multiplier', default=1.0, type=float,
                    help="The width multiplier (only) for ResNet-56 and VGG16-linear. "
                         "Unavailable for other networks. (default 1.0)")
parser.add_argument('--debug', action='store_true',
                    help='Debug mode.')
parser.add_argument('--q_factor', type=float, default=1e-5,
                    help='decay factor (default: 5e-4)')
parser.add_argument('--bin_mode', default=2, type=int, 
                    help='Setup location of bins.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--bias-decay-mult', type=int, default=1,
                    help='Apply bias decay on BatchNorm layers')
parser.add_argument('--log-scale', action='store_true',
                    help='use log scale')
parser.add_argument('--stages', type=int, default=4, 
                    help='number of stages to train (default: 4, single round of training)')
parser.add_argument('--start-stage', default=0, type=int,
                    help='manual stage number (useful on restarts)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss = LossType.from_string(args.loss)
args.decay_epoch = sorted([int(args.epochs * i if i < 1 else i) for i in args.decay_epoch])
if not args.seed:
    args.seed = random.randint(500, 1000)

if args.retrain:
    if not os.path.exists(args.retrain) or not os.path.isfile(args.retrain):
        raise ValueError(f"Path error: {args.retrain}")

if args.clamp != 1.0 and (args.loss == LossType.L1_SPARSITY_REGULARIZATION or args.loss == LossType.ORIGINAL):
    print("WARNING: Clamp only available at Polarization!")

if args.fix_gate:
    if not args.gate:
        raise ValueError("--fix-gate should be with --gate.")

if args.flops_weighted:
    if args.arch not in {'resnet56', 'vgg16_linear'}:
        raise ValueError(f"Unsupported architecture {args.arch}")

if not args.flops_weighted and (args.weight_max is not None or args.weight_min is not None):
    raise ValueError("When --flops-weighted is not set, do not specific --max-weight or --min-weight")

if args.flops_weighted and (args.weight_max is None or args.weight_min is None):
    raise ValueError("When --flops-weighted is set, do specific --max-weight or --min-weight")

if args.max_backup is not None:
    if args.max_backup <= 0:
        raise ValueError("--max-backup is supposed to be greater than 0, got {}".format(args.max_backup))
    pass

if args.target_flops and args.loss != LossType.POLARIZATION:
    raise ValueError(f"Conflict option: --loss {args.loss} --target-flops {args.target_flops}")

if args.target_flops and not args.gate:
    raise ValueError(f"Conflict option: --target-flops only available at --gate mode")

#print(args)
#print(f"Current git hash: {common.get_git_id()}")

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if not os.path.exists(args.save):
    os.makedirs(args.save)
if args.backup_path is not None and not os.path.exists(args.backup_path):
    os.makedirs(args.backup_path)
if not os.path.exists(args.log):
    os.makedirs(args.log)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', download=True, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

num_classes = 10 if args.dataset == 'cifar10' else 100

if not args.retrain:
    if re.match("resnet[0-9]+", args.arch):
        model = models.__dict__[args.arch](num_classes=num_classes,
                                           gate=args.gate,
                                           bn_init_value=args.bn_init_value, aux_fc=False,
                                           width_multiplier=args.width_multiplier,
                                           use_input_mask=args.input_mask)
    elif re.match("vgg[0-9]+", args.arch):
        model = models.__dict__[args.arch](num_classes=num_classes,
                                           gate=args.gate,
                                           bn_init_value=args.bn_init_value,
                                           width_multiplier=args.width_multiplier)
        pass
    else:
        raise NotImplementedError("Do not support {}".format(args.arch))

else:  # initialize model for retraining with configs
    checkpoint = torch.load(args.retrain)
    if args.arch == "resnet56":
        model = models.resnet_expand.resnet56(cfg=checkpoint['cfg'], num_classes=num_classes,
                                              aux_fc=False)
        # initialize corresponding masks
        if "bn3_masks" in checkpoint:
            bn3_masks = checkpoint["bn3_masks"]
            bottleneck_modules = list(filter(lambda m: isinstance(m[1], BasicBlock), model.named_modules()))
            assert len(bn3_masks) == len(bottleneck_modules)
            for i, (name, m) in enumerate(bottleneck_modules):
                if isinstance(m, BasicBlock):
                    if isinstance(m.expand_layer, Identity):
                        continue
                    mask = bn3_masks[i]
                    assert mask[1].shape[0] == m.expand_layer.idx.shape[0]
                    m.expand_layer.idx = np.argwhere(mask[1].clone().cpu().numpy()).squeeze().reshape(-1)
        else:
            raise NotImplementedError("Key bn3_masks expected in checkpoint.")

    elif args.arch == "vgg16_linear":
        model = models.__dict__[args.arch](num_classes=num_classes, cfg=checkpoint['cfg'])
    else:
        raise NotImplementedError(f"Do not support {args.arch} for retrain.")

#training_flops = compute_conv_flops(model, cuda=True)
#print(f"Training model. FLOPs: {training_flops:,}")


def compute_flops_weight(cuda=False):
    # compute the flops weight for each layer in advance
    print("Computing the FLOPs weight...")
    flops_weight = model.compute_flops_weight(cuda=cuda)
    flops_weight_string_builder: typing.List[str] = []
    for fw in flops_weight:
        flops_weight_string_builder.append(",".join(str(w) for w in fw))
    flops_weight_string = "\n".join(flops_weight_string_builder)
    print("FLOPs weight:")
    print(flops_weight_string)
    print()

    return flops_weight_string


if args.flops_weighted:
    flops_weight_string = compute_flops_weight(cuda=True)

if args.cuda:
    model.cuda()

# build optim
if args.bn_wd:
    no_wd_type = [models.common.SparseGate]
else:
    # do not apply weight decay on bn layers
    no_wd_type = [models.common.SparseGate, nn.BatchNorm2d, nn.BatchNorm1d]

no_wd_params = []  # do not apply weight decay on these parameters
for module_name, sub_module in model.named_modules():
    for t in no_wd_type:
        if isinstance(sub_module, t):
            for param_name, param in sub_module.named_parameters():
                if not isinstance(sub_module, models.common.SparseGate): continue
                no_wd_params.append(param)
                #print(f"No weight decay param: module {module_name} param {param_name}")

no_wd_params_set = set(no_wd_params)  # apply weight decay on the rest of parameters
wd_params = []
for param_name, model_p in model.named_parameters():
    if model_p not in no_wd_params_set:
        wd_params.append(model_p)
        #print(f"Weight decay param: parameter name {param_name}")

optimizer = torch.optim.SGD([{'params': list(no_wd_params), 'weight_decay': 0.},
                             {'params': list(wd_params), 'weight_decay': args.weight_decay}],
                            args.lr,
                            momentum=args.momentum)

if args.debug:
    # fake polarization to test pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.BatchNorm2d) or \
                isinstance(module, models.common.SparseGate):
            module.weight.data.zero_()
            total_weight_count = len(module.weight)
            one_num = random.randint(3, total_weight_count - 2)
            module.weight.data[:one_num] = 1.

            print(f"{name} remains {one_num}")

args.mask_list = []
args.stage = 0

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        # reinitialize model with resumed config
        if "vgg" in args.arch and 'cfg' in checkpoint:
            model = models.__dict__[args.arch](num_classes=num_classes,
                                               bn_init_value=args.bn_init_value,
                                               gate=args.gate)
            if args.cuda:
                model.cuda()

        args.start_epoch = 0#checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(checkpoint,'mask_list'):
            args.mask_list = checkpoint['mask_list']
            args.start_stage = checkpoint['stage']

        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
else:
    checkpoint = None
    
if args.loss in {LossType.PROGRESSIVE_SHRINKING,
                 LossType.LOG_QUANTIZATION}:
    teacher_model = copy.deepcopy(model)
    
if not args.loss in {LossType.LOG_QUANTIZATION}:args.stages = 1

history_score = np.zeros((args.epochs, 6))


def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    if args.loss == LossType.L1_SPARSITY_REGULARIZATION:
        sparsity = args.lbd
        bn_modules = list(filter(lambda m: (isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.BatchNorm1d)),
                                 model.named_modules()))
        bn_modules = list(map(lambda m: m[1], bn_modules))  # remove module name
        for m in bn_modules:
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))
    else:
        raise NotImplementedError(f"Do not support loss: {args.loss}")


def clamp_bn(model, lower_bound=0, upper_bound=1):
    if model.gate:
        sparse_modules = list(filter(lambda m: isinstance(m, SparseGate), model.modules()))
    else:
        sparse_modules = list(
            filter(lambda m: isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d), model.modules()))

    for m in sparse_modules:
        m.weight.data.clamp_(lower_bound, upper_bound)


def set_bn_zero(model: nn.Module, threshold=0.0) -> (nn.Module, int):
    """
    Set bn bias to zero
    Note: The operation is inplace. Parameters of the model will be changed!
    :param model: to set
    :param threshold: set bn bias to zero if corresponding lambda <= threshold
    :return modified model, the number of zero bn channels
    """
    with torch.no_grad():
        mask_length = 0
        for name, sub_module in model.named_modules():
            # only process bn modules
            if not (isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d)):
                continue

            mask = sub_module.weight.detach() <= threshold
            sub_module.weight[mask] = 0.
            sub_module.bias[mask] = 0.

            mask_length += torch.sum(mask).item()

    return model, mask_length


def bn_sparsity(model, loss_type, sparsity, t, alpha,
                flops_weighted: bool, weight_min=None, weight_max=None):
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    """
    bn_modules = model.get_sparse_layers()

    if loss_type == LossType.POLARIZATION or loss_type == LossType.L2_POLARIZATION:
        # compute global mean of all sparse vectors
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))
        sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_

        sparsity_loss = 0.
        if flops_weighted:
            for sub_module in model.modules():
                if isinstance(sub_module, model.building_block):
                    flops_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                    sub_module_sparse_layers = sub_module.get_sparse_modules()

                    for sparse_m, flops_w in zip(sub_module_sparse_layers, flops_weight):
                        # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]
                        flops_w = weight_min + (weight_max - weight_min) * flops_w

                        sparsity_term = t * torch.sum(torch.abs(sparse_m.weight.view(-1))) - torch.sum(
                            torch.abs(sparse_m.weight.view(-1) - alpha * sparse_weights_mean))
                        sparsity_loss += flops_w * sparsity * sparsity_term
            return sparsity_loss
        else:
            for m in bn_modules:
                if loss_type == LossType.POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        torch.abs(m.weight - alpha * sparse_weights_mean))
                elif loss_type == LossType.L2_POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        (m.weight - alpha * sparse_weights_mean) ** 2)
                else:
                    raise ValueError(f"Unexpected loss type: {loss_type}")
                sparsity_loss += sparsity * sparsity_term

            return sparsity_loss
    else:
        raise ValueError()
    
def prune_by_thresh(model,left=0,right=100):
    import copy
    pruned_model = copy.deepcopy(model)
        
    bn_modules = pruned_model.get_sparse_layers()
    
    ch_start = 0
    for bn_module in bn_modules:
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            mask0 = bn_module.weight.data.abs()<left
            mask1 = bn_module.weight.data.abs()>right
            inactive = torch.logical_or(mask0,mask1)
            bn_module.weight.data[inactive] = 0
            ch_start += ch_len
    return pruned_model
    
def mean_sparsity(x,sf_split,sparse_coef=None,N=None):
    order = 1
    if order == 1:
        grad = (args.t + sparse_coef - torch.sign(x-sf_split))
        x -= args.lbd * args.current_lr * grad
    else:
        grad = args.t - 2.*(N-1)*(N-1)/N/N*x + 2.*(N-1)/N*(sf_split*N-x)/N + 2./N * (sf_split*N-x-sf_split*(N-1))
        x -= args.lbd * grad * args.current_lr
    return x
        
args.eps = 1e-10
        
# assign bin indices to scale factors
# num_bins: number of total bins
# target_indices: indices of bins that need to be assigned
# default_index: those not assigned to target indices will be assigned the default index 
# num_bins and target_indices can be adjust to get any ratio
def assign_to_indices(bn_modules):
    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors,torch.abs(bn_module.weight.data)))
    
    # total channels
    total_channels = len(all_scale_factors)
    ch_per_bin = total_channels//args.stages
    shrink = torch.ones(total_channels).long().cuda()
    targeted = torch.zeros(total_channels).long().cuda()
    
    # do not sort masked channels
    for mask in args.mask_list[:args.current_stage]:
        if mask is not None:
            shrink[mask==1] = 0
    not_assigned = shrink.nonzero()
    remain_factors = all_scale_factors[not_assigned] 
    tmp,ch_indices = remain_factors.sort(dim=0)
    # keep the most important and remaining bin
    selected = not_assigned[ch_indices[-ch_per_bin:]]
    shrink[selected] = 0
    targeted[selected] = 1
    
    return shrink,targeted
    
def sample_network(old_model,net_id=None,zero_bias=True,eval=False):
    if net_id is None:
        net_id = torch.tensor(0).random_(0,4)
    all_scale_factors = torch.tensor([]).cuda()
    if eval:
        old_model = copy.deepcopy(old_model)
        
    bn_modules = old_model.get_sparse_layers()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors,bn_module.weight.data))
    
    # total channels
    total_channels = len(all_scale_factors)
    channel_per_layer = total_channels//4
    
    _,ch_indices = all_scale_factors.sort(dim=0)
    
    weight_valid_mask = torch.zeros(total_channels).long().cuda()
    weight_valid_mask[ch_indices[channel_per_layer*(3-net_id):]] = 1
        
    if True:
        freeze_mask = torch.ones(total_channels).long().cuda()
        freeze_mask[ch_indices[channel_per_layer*(3-net_id):channel_per_layer*(4-net_id)]] = 0
    else:
        freeze_mask = 1-weight_valid_mask
    ch_start = 0
    for bn_module in bn_modules:
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            inactive = weight_valid_mask[ch_start:ch_start+ch_len]==0
            bn_module.weight.data[inactive] = 0
            if zero_bias:
                bn_module.bias.data[inactive] = 0
            ch_start += ch_len
    if not eval:
        return freeze_mask
    else:
        return test(old_model)
        
def prune_by_mask(old_model,mask_list,zero_bias=True):
    import copy
    pruned_model = copy.deepcopy(old_model)
        
    bn_modules = pruned_model.get_sparse_layers()
    
    tokeep = None
    for mask in mask_list:
        if tokeep is None:
            tokeep = mask.clone().detach()
        else:
            tokeep += mask.clone().detach()
        
    ch_start = 0
    for bn_module in bn_modules:
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            inactive = tokeep[ch_start:ch_start+ch_len]==0
            bn_module.weight.data[inactive] = 0
            if zero_bias:
                bn_module.bias.data[inactive] = 0
            ch_start += ch_len
    #for name, param in model.named_parameters(): print(name, param.data)
    return pruned_model
   
def recover_weights(new_model,old_model,mask_list):
    bns1,convs1 = new_model.get_sparse_layers_and_convs()
    bns2,convs2 = old_model.get_sparse_layers_and_convs()
    ch_start = 0
    for conv1,bn1,conv2,bn2 in zip(convs1,bns1,convs2,bns2):
        ch_len = conv1.weight.data.size(0)
        for freeze_mask in mask_list:
            if freeze_mask is None:continue
            with torch.no_grad():
                freeze_mask = freeze_mask[ch_start:ch_start+ch_len] == 1
                bn1.weight.data[freeze_mask] = bn2.weight.data[freeze_mask].clone().detach()
                bn1.running_mean.data[freeze_mask] = bn2.running_mean.data[freeze_mask].clone().detach()
                bn1.running_var.data[freeze_mask] = bn2.running_var.data[freeze_mask].clone().detach()
                if hasattr(bn1, 'bias') and bn1.bias is not None:
                    bn1.bias.data[freeze_mask] = bn2.bias.data[freeze_mask].clone().detach()
                if isinstance(conv1, nn.Conv2d):
                    conv1.weight.data[freeze_mask, :, :, :] = conv2.weight.data[freeze_mask, :, :, :].clone().detach()
                else:
                    conv1.weight.data[freeze_mask, :] = conv2.weight.data[freeze_mask, :].clone().detach()
                if hasattr(conv1, 'bias') and conv1.bias is not None:
                    conv1.bias.data[freeze_mask] = conv2.bias.data[freeze_mask].clone().detach()
        ch_start += ch_len
    
    if args.current_stage >= 1:
        new_model.conv1.weight.data = old_model.conv1.weight.data.clone().detach()
        new_model.bn1.weight.data = old_model.bn1.weight.data.clone().detach()
        new_model.bn1.bias.data = old_model.bn1.bias.data.clone().detach()
        new_model.bn1.running_mean.data = old_model.bn1.running_mean.data.clone().detach()
        new_model.bn1.running_var.data = old_model.bn1.running_var.data.clone().detach()
        new_model.linear.weight.data = old_model.linear.weight.data.clone().detach()
        new_model.linear.bias.data = old_model.linear.bias.data.clone().detach()
            
def compare_models(old,new):
    #for name, param in new.named_parameters(): print(name, param.size())
    #exit(0)
    bns1,convs1 = old.get_sparse_layers_and_convs()
    bns2,convs2 = new.get_sparse_layers_and_convs()
    ch_start = 0
    for conv1,bn1,conv2,bn2 in zip(convs1,bns1,convs2,bns2):
        ch_len = conv1.weight.data.size(0)
        for freeze_mask in args.mask_list[:args.current_stage]:
            if freeze_mask is None:continue
            freeze_mask = freeze_mask[ch_start:ch_start+ch_len] == 1
            assert torch.equal(conv1.weight.data[freeze_mask, :, :, :], conv2.weight.data[freeze_mask, :, :, :])
            assert torch.equal(bn1.weight.data[freeze_mask], bn2.weight.data[freeze_mask])
            assert torch.equal(bn1.bias.data[freeze_mask], bn2.bias.data[freeze_mask])
            assert torch.equal(bn1.running_mean.data[freeze_mask],bn2.running_mean.data[freeze_mask])
            assert torch.equal(bn1.running_var.data[freeze_mask],bn2.running_var.data[freeze_mask])
            #assert torch.equal(conv1.weight.data, conv2.weight.data)
            #assert torch.equal(bn1.weight.data, bn2.weight.data)
            #assert torch.equal(bn1.bias.data, bn2.bias.data)
        ch_start += ch_len
        
def log_quantization(old_model):
    if args.current_stage == args.stages - 1:
        return
        
    bn_modules = old_model.get_sparse_layers()
    shrink,targeted = assign_to_indices(bn_modules)
    # update mask of current stage
    if len(args.mask_list) < args.current_stage+1:
        args.mask_list.append(targeted.clone().detach())
    else:
        args.mask_list[-1] = targeted.clone().detach()
    
    ch_start = 0
    for bn_module in bn_modules:
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            shrink_mask = shrink[ch_start:ch_start+ch_len] == 1
            bn_module.weight.data[shrink_mask] -= args.lbd * args.current_lr * 400
            if args.weight_decay!=1:
                bn_module.bias.data[shrink_mask] *= 1 - args.current_lr * args.weight_decay * args.bias_decay_mult
            ch_start += ch_len
    
def factor_visualization(iter, model, prec):
    scale_factors = torch.tensor([]).cuda()
    biases = torch.tensor([]).cuda()
    bn_modules = model.get_sparse_layers()
    for bn_module in bn_modules:
        scale_factors = torch.cat((scale_factors,(bn_module.weight.data.view(-1))))
        biases = torch.cat((biases,(bn_module.bias.data.view(-1))))
    # plot figure
    save_dir = args.save + 'factor/' + str(args.current_stage) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig, axs = plt.subplots(ncols=4, figsize=(20,4))
    # plots
    sns.histplot(scale_factors.detach().cpu().numpy(), ax=axs[0])
    sns.histplot(torch.log10(torch.clamp(scale_factors.abs(),min=args.eps)).detach().cpu().numpy(), ax=axs[1])

    sns.histplot(biases.detach().cpu().numpy(), ax=axs[2])
    sns.histplot(torch.log10(torch.clamp(biases.abs(),min=args.eps)).detach().cpu().numpy(), ax=axs[3])
    fig.savefig(save_dir + f'{iter:03d}_{prec:.3f}.png')
    plt.close('all')
        

def prune_while_training(model: nn.Module, arch: str, prune_mode: str, num_classes: int):
    target_ratios = []#.25,.5,.75]#[0.1 + 0.1*x for x in range(9)]
    saved_flops = []
    saved_prec1s = []
    saved_thresh = []
    if arch == "resnet56":
        from resprune_gate import prune_resnet
        from models.resnet_expand import resnet56 as resnet50_expand
        for strat in []:#['grad','fixed']:
            saved_model,thresh = prune_resnet(sparse_model=model, pruning_strategy=strat,
                                            sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
            prec1 = test(saved_model.cuda())
            flop = compute_conv_flops(saved_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
            saved_thresh += [thresh]
        for ratio in target_ratios:
            saved_model,thresh = prune_resnet(sparse_model=model, pruning_strategy='fixed', prune_type='ns', l1_norm_ratio=ratio,
                                             sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
            prec1 = test(saved_model.cuda())
            flop = compute_conv_flops(saved_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
            saved_thresh += [thresh]
    elif arch == 'vgg16_linear':
        from vggprune_gate import prune_vgg
        from models import vgg16_linear
        # todo: update
        for ratio in target_ratios:
            saved_model,thresh = prune_vgg(sparse_model=model, pruning_strategy='fixed', prune_type='ns', l1_norm_ratio=ratio,
                                          sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
            prec1 = test(saved_model.cuda())
            flop = compute_conv_flops(saved_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
            saved_thresh += [thresh]
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")

    baseline_flops = compute_conv_flops(model, cuda=True)
        
    inplace_precs = []
    if args.loss in {LossType.LOG_QUANTIZATION}:
        for i in range(min(3,len(args.mask_list))):
            inplace_precs += [test(prune_by_mask(model,args.mask_list[:i+1],zero_bias=True))]
            inplace_precs += [test(prune_by_mask(model,args.mask_list[:i+1],zero_bias=False))]
    
    if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
        for i in range(0, 3):
            inplace_precs += [sample_network(model,net_id=i,zero_bias=True,eval=True)]
        
    
    print_str = ''
    for flop,prec1,thresh in zip(saved_flops,saved_prec1s,saved_thresh):
        print_str += f"[{prec1:.4f}({flop / baseline_flops:.4f}), {thresh:.10f}]\t"
        
    for prec1 in inplace_precs:
        print_str += f"{prec1:.4f}\t"
        
    print(print_str)

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

def train(epoch):
    model.train()
    global history_score, global_step
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0
    train_iter = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_iter):
        if args.loss in {LossType.LOG_QUANTIZATION,
                         LossType.PROGRESSIVE_SHRINKING}:
            old_model = copy.deepcopy(model)
        if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
            freeze_mask = sample_network(model)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)
        if False and args.loss in {LossType.PROGRESSIVE_SHRINKING,
                         LossType.LOG_QUANTIZATION}:
            soft_logits = teacher_model(data)
            if isinstance(soft_logits, tuple):
                soft_logits, _ = soft_logits
            soft_label = F.softmax(soft_logits.detach(), dim=1)
            loss = cross_entropy_loss_with_soft_target(output, soft_label)

        # logging
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_data += target.data.shape[0]

        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            sparsity_loss = bn_sparsity(model, args.loss, args.lbd,
                                        t=args.t, alpha=args.alpha,
                                        flops_weighted=args.flops_weighted,
                                        weight_max=args.weight_max, weight_min=args.weight_min)
            loss += sparsity_loss
            avg_sparsity_loss += sparsity_loss.data.item()
        loss.backward()
        if args.loss in {LossType.L1_SPARSITY_REGULARIZATION}:
            updateBN()
        if args.loss in {LossType.LOG_QUANTIZATION}:
            log_quantization(model)
        optimizer.step()
        if args.loss in {LossType.LOG_QUANTIZATION}:
            recover_weights(model,old_model,args.mask_list[:args.current_stage])
        if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
            recover_weights(model,old_model,[freeze_mask])
        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION,
                         LossType.LOG_QUANTIZATION,
                         LossType.PROGRESSIVE_SHRINKING}:
            clamp_bn(model, upper_bound=args.clamp)
        global_step += 1
        train_iter.set_description(
            'Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]. Loss: {:.6f}'.format(
            global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), avg_loss / len(train_loader)))
                                
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = float(train_acc) / float(total_data)
    history_score[epoch][3] = avg_sparsity_loss / len(train_loader)
    pass


def test(modelx):
    modelx.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = modelx(data)
            if isinstance(output, tuple):
                output, output_aux = output
            test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, backup: bool, backup_path: str, epoch: int, max_backup: int):
    state['args'] = args

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    if backup and backup_path is not None:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))

        if max_backup is not None:
            while True:
                # remove redundant backup checkpoints to save space
                checkpoint_match = map(lambda f_name: re.fullmatch("checkpoint_([0-9]+).pth.tar", f_name),
                                       os.listdir(backup_path))
                checkpoint_match = filter(lambda m: m is not None, checkpoint_match)
                checkpoint_id: typing.List[int] = list(map(lambda m: int(m.group(1)), checkpoint_match))
                checkpoint_count = len(checkpoint_id)
                if checkpoint_count > max_backup:
                    min_checkpoint_epoch = min(checkpoint_id)
                    min_checkpoint_path = os.path.join(backup_path,
                                                       'checkpoint_{}.pth.tar'.format(min_checkpoint_epoch))
                    print(f"Too much checkpoints (Max {max_backup}, got {checkpoint_count}).")
                    print(f"Remove file: {min_checkpoint_path}")
                    os.remove(min_checkpoint_path)
                else:
                    break


best_prec1 = 0.
global_step = 0
prec1_list = []

if args.evaluate:
    prec1 = test(model)
    print(f"All Prec1: {prec1}")
    factor_visualization(0, model, prec1)
    prune_while_training(model, arch=args.arch,
                       prune_mode="default",
                       num_classes=num_classes)
         

for args.current_stage in range(args.start_stage, args.stages):
    # init non-freezing weights
    if args.loss in {LossType.LOG_QUANTIZATION} and args.current_stage >= 1:
        old_model = copy.deepcopy(model)
        model._initialize_weights(1.0)
        recover_weights(model,old_model,args.mask_list[:args.current_stage])
    for epoch in range(args.start_epoch, args.epochs):
        if args.max_epoch is not None and epoch >= args.max_epoch:
            break

        args.current_lr = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
        print("Start epoch {}/{} stage {}/{} with learning rate {}...".format(epoch, args.epochs, args.current_stage, args.stages, args.current_lr))

        train(epoch) # train with regularization

        prec1 = test(model)
        print(f"All Prec1: {prec1}")
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': prec1,
            'optimizer': optimizer.state_dict(),
            'mask_list': args.mask_list,
            'stage': args.current_stage,
        }, is_best, filepath=args.save,
            backup_path=args.backup_path,
            backup=epoch % args.backup_freq == 0,
            epoch=epoch,
            max_backup=args.max_backup
        )
        
        # visualize scale factors
        #factor_visualization(epoch, model, prec1)

        # flops
        prune_while_training(model, arch=args.arch,prune_mode="default",num_classes=num_classes)
    print("Best accuracy: " + str(best_prec1))
    prec1_list += [prec1]
print(prec1_list)