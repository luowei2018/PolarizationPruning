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
parser.add_argument('--q_factor', type=float, default=0.0001,
                    help='decay factor (default: 0.001)')
parser.add_argument('--bin_mode', default=2, type=int, 
                    help='Setup location of bins.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

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

print(args)
print(f"Current git hash: {common.get_git_id()}")

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

training_flops = compute_conv_flops(model, cuda=True)
print(f"Training model. FLOPs: {training_flops:,}")


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


def freeze_sparse_gate(model: nn.Module):
    # do not update all SparseGate
    for sub_module in model.modules():
        if isinstance(sub_module, models.common.SparseGate):
            for p in sub_module.parameters():
                # do not update SparseGate
                p.requires_grad = False


if args.fix_gate:
    if args.lbd != 0:
        raise ValueError("The lambda must be 0 in fix-gate mode.")
    # do not update all SparseGate
    freeze_sparse_gate(model)

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
                no_wd_params.append(param)
                print(f"No weight decay param: module {module_name} param {param_name}")

no_wd_params_set = set(no_wd_params)  # apply weight decay on the rest of parameters
wd_params = []
for param_name, model_p in model.named_parameters():
    if model_p not in no_wd_params_set:
        wd_params.append(model_p)
        print(f"Weight decay param: parameter name {param_name}")

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

        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
else:
    checkpoint = None

history_score = np.zeros((args.epochs, 6))


def bn_weights(model):
    weights = []
    bias = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weights.append((name, m.weight.data))
            bias.append((name, m.bias.data))

    return weights, bias
    pass


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
        
def log_quantization(model):
    #############SETUP###############
    args.ista_err = torch.tensor([0.0]).cuda(0)
    # locations of bins should fit original dist
    # start can be tuned to find a best one
    # distance between bins min=2
    if args.bin_mode ==2:
        num_bins, bin_start, bin_stride = 4, -6, 2
    elif args.bin_mode == 1:
        num_bins, bin_start, bin_stride = 6, -5, 1
    else:
        print("Bin mode not supported")
        exit(1)
    # how centralize the bin is, relax this may improve prec
    bin_width = 1e-1
    # locations we want to quantize
    args.bins = torch.pow(10.,torch.tensor([bin_start+bin_stride*x for x in range(num_bins)])).cuda(0)
    # trade-off of original distribution and new distribution
    # big: easy to get new distribution, but may degrade performance
    # small: maintain good performance but may not affect distribution much
    decay_factor = args.q_factor # lower this to improve perf
    # how small/low rank bins get more advantage
    amp_factors = torch.tensor([2**(num_bins-1-x) for x in range(num_bins)]).cuda()
    args.ista_err_bins = [0 for _ in range(num_bins)]
    args.ista_cnt_bins = [0 for _ in range(num_bins)]
    
    #################START###############
    def get_min_idx(x):
        args.bins = torch.pow(10.,torch.tensor([bin_start+bin_stride*x for x in range(num_bins)])).to(x.device)
        dist = torch.abs(torch.log10(torch.abs(x).unsqueeze(-1)/args.bins))
        _,min_idx = dist.min(dim=-1)
        return min_idx
        
    def get_bin_distribution(x):
        x = torch.clamp(torch.abs(x), min=1e-8) * torch.sign(x)
        min_idx = get_min_idx(x)
        all_err = torch.log10(args.bins[min_idx]/torch.abs(x))
        abs_err = torch.abs(all_err)
        # calculate total error
        args.ista_err += abs_err.sum()
        # calculating err for each bin
        for i in range(num_bins):
            if torch.sum(min_idx==i)>0:
                args.ista_err_bins[i] += abs_err[min_idx==i].sum().cpu().item()
                args.ista_cnt_bins[i] += torch.numel(abs_err[min_idx==i])
                
    def redistribute(x,bin_indices):
        tar_bins = args.bins[bin_indices]
        # amplifier based on rank of bin
        amp = amp_factors[bin_indices]
        all_err = torch.log10(tar_bins/torch.abs(x))
        abs_err = torch.abs(all_err)
        # more distant larger multiplier
        # pull force relates to distance and target bin (how off-distribution is it?)
        # low rank bin gets higher pull force
        distance = torch.log10(tar_bins/torch.abs(x))
        multiplier = 10**(distance*decay_factor*amp)
        x[abs_err>bin_width] *= multiplier[abs_err>bin_width]
        return x
        
    bn_modules = model.get_sparse_layers()
    
    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        with torch.no_grad():
            get_bin_distribution(bn_module.weight.data)
        all_scale_factors = torch.cat((all_scale_factors,torch.abs(bn_module.weight.data)))
    # total channels
    total_channels = len(all_scale_factors)
    ch_per_bin = total_channels//num_bins
    _,bin_indices = torch.tensor(args.ista_cnt_bins).sort()
    remain = torch.ones(total_channels).long().cuda()
    assigned_binindices = torch.zeros(total_channels).long().cuda()
    
    for bin_idx in bin_indices[:-1]:
        dist = torch.abs(torch.log10(args.bins[bin_idx]/all_scale_factors)) 
        not_assigned = remain.nonzero()
        # remaining channels importance
        chan_imp = dist[not_assigned] 
        tmp,ch_indices = chan_imp.sort(dim=0)
        selected_in_remain = ch_indices[:ch_per_bin]
        selected = not_assigned[selected_in_remain]
        remain[selected] = 0
        assigned_binindices[selected] = bin_idx
    assigned_binindices[remain.nonzero()] = bin_indices[-1]
        
    ch_start = 0
    for bn_module in bn_modules:
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            bn_module.weight.data = redistribute(bn_module.weight.data, assigned_binindices[ch_start:ch_start+ch_len])
            ch_start += ch_len
        
    
    
def factor_visualization(iter, model, prec):
    scale_factors = torch.tensor([]).cuda()
    biases = torch.tensor([]).cuda()
    bn_modules = model.get_sparse_layers()
    for bn_module in bn_modules:
        scale_factors = torch.cat((scale_factors,torch.abs(bn_module.weight.data.view(-1))))
        biases = torch.cat((biases,torch.abs(bn_module.bias.data.view(-1))))
    # plot figure
    save_dir = args.save + 'factor/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig, axs = plt.subplots(ncols=4, figsize=(20,4))
    # plots
    sns.histplot(scale_factors.detach().cpu().numpy(), ax=axs[0])
    scale_factors = torch.clamp(scale_factors,min=1e-10)
    sns.histplot(torch.log10(scale_factors).detach().cpu().numpy(), ax=axs[1])
    
    sns.histplot(biases.detach().cpu().numpy(), ax=axs[2])
    biases = torch.clamp(biases,min=1e-10)
    sns.histplot(torch.log10(biases).detach().cpu().numpy(), ax=axs[3])
    fig.savefig(save_dir + f'{iter:03d}_{prec:.3f}.png')
    plt.close('all')
        

def prune_while_training(model: nn.Module, arch: str, prune_mode: str, num_classes: int):
    target_ratios = [0.1 + 0.1*x for x in range(9)]
    saved_flops = []
    saved_prec1s = []
    if arch == "resnet56":
        from resprune_gate import prune_resnet
        from models.resnet_expand import resnet56 as resnet50_expand
        for ratio in target_ratios:
            saved_model = prune_resnet(sparse_model=model, pruning_strategy='fixed', prune_type='ns', l1_norm_ratio=ratio,
                                             sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
            prec1 = test(saved_model.cuda())
            flop = compute_conv_flops(saved_model, cuda=True)
            saved_prec1s += [prec1]
            saved_flops += [flop]
        baseline_model = resnet50_expand(num_classes=num_classes, gate=False, aux_fc=False)
    elif arch == 'vgg16_linear':
        from vggprune_gate import prune_vgg
        from models import vgg16_linear

        saved_model_grad = prune_vgg(num_classes=num_classes, sparse_model=model, prune_mode=prune_mode,
                                     sanity_check=False, pruning_strategy='grad')
        saved_model_fixed = prune_vgg(num_classes=num_classes, sparse_model=model, prune_mode=prune_mode,
                                      sanity_check=False, pruning_strategy='fixed')
        baseline_model = vgg16_linear(num_classes=num_classes, gate=False)
    else:
        # not available
        raise NotImplementedError(f"do not support arch {arch}")

    baseline_flops = compute_conv_flops(baseline_model, cuda=True)
    
    for flop,prec1 in zip(saved_flops,saved_prec1s):
        print(f" --> FLOPs : {flop:,}, ratio: {flop / baseline_flops}, prec1: {prec1}")


def train(epoch):
    model.train()
    global history_score, global_step
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0
    train_iter = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_iter):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output, output_aux = output
        loss = F.cross_entropy(output, target)

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
        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            clamp_bn(model, upper_bound=args.clamp)
        global_step += 1
        if batch_idx % args.log_interval == 0:
            if args.loss not in {LossType.LOG_QUANTIZATION}:
                train_iter.set_description(
                    'Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]. Loss: {:.6f}'.format(
                    global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader), loss.data.item()))
            else:
                ista_err = args.ista_err.cpu().item()
                train_iter.set_description(
                    'Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]. Loss: {:.6f}. ISTA-Err: {:.4f}'.format(
                    global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader), loss.data.item(), ista_err))

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

writer = SummaryWriter(logdir=args.log)

if args.flops_weighted:
    writer.add_text("train/conv_flops_weight", flops_weight_string, global_step=0)

if args.evaluate:
    prec1 = test(model)
    print(f"All Prec1: {prec1}")
    factor_visualization(0, model, prec1)
    prune_while_training(model, arch=args.arch,
                       prune_mode="default",
                       num_classes=num_classes)
    exit(0)

for epoch in range(args.start_epoch, args.epochs):
    if args.max_epoch is not None and epoch >= args.max_epoch:
        break

    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
    print("Start epoch {}/{} with learning rate {}...".format(epoch, args.epochs, current_learning_rate))

    weights, bias = bn_weights(model)
    for bn_name, bn_weight in weights:
        writer.add_histogram("bn/" + bn_name, bn_weight, global_step=epoch)
    for bn_name, bn_bias in bias:
        writer.add_histogram("bn_bias/" + bn_name, bn_bias, global_step=epoch)
    # visualize conv kernels
    for name, sub_modules in model.named_modules():
        if isinstance(sub_modules, nn.Conv2d):
            writer.add_histogram("conv_kernels/" + name, sub_modules.weight, global_step=epoch)
    if args.gate:
        for gate_name, m in model.named_modules():
            if isinstance(m, SparseGate):
                writer.add_histogram("gate/" + gate_name, m.weight, global_step=epoch)

    train(epoch) # train with regularization

    prec1 = test(model)
    print(f"All Prec1: {prec1}")
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save,
        backup_path=args.backup_path,
        backup=epoch % args.backup_freq == 0,
        epoch=epoch,
        max_backup=args.max_backup
    )
    
    # visualize scale factors
    factor_visualization(epoch, model, prec1)

    # write the tensorboard
    writer.add_scalar("train/average_loss", history_score[epoch][0], epoch)
    writer.add_scalar("train/sparsity_loss", history_score[epoch][3], epoch)
    writer.add_scalar("train/train_acc", history_score[epoch][1], epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar("val/acc", prec1, epoch)
    writer.add_scalar("val/best_acc", best_prec1, epoch)

    # flops
    # peek the remaining flops
    prune_while_training(model, arch=args.arch,
                       prune_mode="default",
                       num_classes=num_classes)
    
    # show log quantization result
    if args.loss in {LossType.LOG_QUANTIZATION}:
        print('BinErr:', " ".join(format(x, ".3f") for x in args.ista_err_bins))
        print('BinCnt:', " ".join(format(x, "05d") for x in args.ista_cnt_bins), args.bins)

if args.loss == LossType.POLARIZATION and args.target_flops and (
        flops_grad / baseline_flops) > args.target_flops and args.gate:
    print("WARNING: the FLOPs does not achieve the target FLOPs at the end of training.")
print("Best accuracy: " + str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt='%10.5f', delimiter=',')

writer.close()

print("Best accuracy: " + str(best_prec1))
