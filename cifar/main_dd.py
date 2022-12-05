# 11 version
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
from torchvision import datasets, transforms

import common
import models
from common import LossType, compute_conv_flops
from models.common import SparseGate, Identity
from models.resnet_expand import BasicBlock
import copy
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
parser.add_argument('--oneshot_ltr', action='store_true',
                    help='If use randomly initialized weights.')
parser.add_argument('--remain_ratio', default=1.0, type=float,
                    help='The remaining network ratio after pruning.')
parser.add_argument('--noise_ratio', default=0.0, type=float,
                    help='The noise added to the label.')
# parser.add_argument('--prune_scale', type=int, default=0,
#                     help='Whether the model is orginal(2) or pruned(1) or training w/ both(0).')
# current default:
parser.add_argument('--alphas', type=float, nargs='+', default=[1, 1],
                    help='Multiplier of each subnet.')
parser.add_argument('--ps_batch', type=int, default=1,
                    help='number of back propagation per weight update')          
parser.add_argument('--pruning_step', type=float, default=0.1,
                    help='iterative pruning step')      

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
        model = models.__dict__[args.arch](num_classes=num_classes)
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


def prune_while_training(model: nn.Module, arch: str, prune_mode: str, num_classes: int):
    if arch == "resnet56":
        from resprune_gate import prune_resnet
        from models.resnet_expand import resnet56 as resnet50_expand
        saved_model_grad = prune_resnet(sparse_model=model, pruning_strategy='grad',
                                        sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
        saved_model_fixed = prune_resnet(sparse_model=model, pruning_strategy='fixed',
                                         sanity_check=False, prune_mode=prune_mode, num_classes=num_classes)
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

    saved_flops_grad = compute_conv_flops(saved_model_grad, cuda=True)
    saved_flops_fixed = compute_conv_flops(saved_model_fixed, cuda=True)
    baseline_flops = compute_conv_flops(baseline_model, cuda=True)

    return saved_flops_grad, saved_flops_fixed, baseline_flops

def create_mask(model, remain_ratio=1.0):
    bn_modules,convs = model.get_sparse_layers_and_convs()
    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors,bn_module.weight.data.abs())) # 之后可以把bn_module.weight.data.abs()换成别的importance

    # total channels
    total_channels = len(all_scale_factors)
    _,ch_indices = all_scale_factors.sort(dim=0)
    
    weight_valid_mask = torch.zeros(total_channels).long().cuda()
    weight_valid_mask[ch_indices[int(total_channels*(1 - remain_ratio)):]] = 1
    return weight_valid_mask

#-#-#-#-#
def iter_create_mask(model, weight_valid_mask, remain_ratio=1.0, pruning_step=0.1):
    bn_modules,convs = model.get_sparse_layers_and_convs()
    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors,bn_module.weight.data.abs()))

    total_channels = len(all_scale_factors)
    if weight_valid_mask is None:
        weight_valid_mask = torch.ones(total_channels).long().cuda()
        return weight_valid_mask
    print(weight_valid_mask)
    new_scale_factors_indices = torch.where(weight_valid_mask==1)
    print(all_scale_factors)
    new_factors = all_scale_factors[new_scale_factors_indices]
    print(new_factors)
    _, new_ch_indices = new_factors.sort(dim=0)

    weight_valid_mask = torch.zeros(total_channels).long().cuda()
    #print("new_scale_factors_indices")
    #print(new_scale_factors_indices)
    #print(len(new_scale_factors_indices[0]))
    print(new_ch_indices[int(len(new_scale_factors_indices[0])*pruning_step):])
    weight_valid_mask[new_scale_factors_indices[0][new_ch_indices[int(len(new_scale_factors_indices[0])*pruning_step):]]] = 1

    return weight_valid_mask

# baseline - original oneshot
def sample_network(model, weight_valid_mask):
    ch_start = 0
    bn_modules,convs = model.get_sparse_layers_and_convs()
    for bn_module,conv in zip(bn_modules,convs):
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            inactive = weight_valid_mask[ch_start:ch_start+ch_len]==0
            bn_module.weight.data[inactive] = 0
            bn_module.bias.data[inactive] = 0
            out_channel_mask = weight_valid_mask[ch_start:ch_start+ch_len]==1
            bn_module.out_channel_mask = out_channel_mask.clone().detach()
            ch_start += ch_len

def sample_network_dynamic(model, weight_valid_mask, separation, eval=False):
    dynamic_model = copy.deepcopy(model)
    for module_name, bn_module in dynamic_model.named_modules():
        if not isinstance(bn_module, nn.BatchNorm2d) and not isinstance(bn_module, nn.BatchNorm1d):
            continue
        bn_module.running_mean.data = bn_module._buffers[f"mean{separation}"]
        bn_module.running_var.data = bn_module._buffers[f"var{separation}"]
    bn_modules,convs = dynamic_model.get_sparse_layers_and_convs()

    all_scale_factors = torch.tensor([]).cuda()
    for bn_module in bn_modules:
        all_scale_factors = torch.cat((all_scale_factors, bn_module.weight.data.abs()))

    total_channels = len(all_scale_factors)
    _,ch_indices = all_scale_factors.sort(dim=0)
    freeze_mask = 1 - weight_valid_mask

    ch_start = 0
    for bn_module,conv in zip(bn_modules,convs):
        with torch.no_grad():
            ch_len = len(bn_module.weight.data)
            inactive = weight_valid_mask[ch_start:ch_start+ch_len]==0
            bn_module.weight.data[inactive] = 0
            bn_module.bias.data[inactive] = 0
            out_channel_mask = weight_valid_mask[ch_start:ch_start+ch_len]==1
            bn_module.out_channel_mask = out_channel_mask.clone().detach()
            ch_start += ch_len

    if not eval:
        return freeze_mask, dynamic_model, ch_indices
    else:
        return dynamic_model

def get_non_sparse_modules(model,get_name=False):
    sparse_modules = []
    bn_modules,conv_modules = model.get_sparse_layers_and_convs()
    for bn,conv in zip(bn_modules,conv_modules):
        sparse_modules.append(bn)
        sparse_modules.append(conv)
    sparse_modules_set = set(sparse_modules)
    non_sparse_modules = []
    for module_name, module in model.named_modules():
        if module not in sparse_modules_set:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
                if not get_name:
                    non_sparse_modules.append(module)
                else:
                    non_sparse_modules.append((module_name,module))
    return non_sparse_modules

def update_shared_model(old_model,new_model,mask,batch_idx,ch_indices,net_id):
    def copy_module_grad(old_module,new_module,subnet_mask=None,enhance_mask=None):
        if subnet_mask is not None:
            freeze_mask = subnet_mask == 1
            keep_mask = subnet_mask == 0

        # copy running mean/var
        if isinstance(new_module,nn.BatchNorm2d) or isinstance(new_module,nn.BatchNorm1d):
            #if args.split_running_stat:
            if subnet_mask is not None:
                old_module._buffers[f"mean{net_id}"][keep_mask] = new_module.running_mean.data[keep_mask].clone().detach()
                old_module._buffers[f"var{net_id}"][keep_mask] = new_module.running_var.data[keep_mask].clone().detach()
            else:
                old_module._buffers[f"mean{net_id}"] = new_module.running_mean.data.clone().detach()
                old_module._buffers[f"var{net_id}"] = new_module.running_var.data.clone().detach()
            # else:
            #     if subnet_mask is not None:
            #         old_module.running_mean.data[keep_mask] = new_module.running_mean.data[keep_mask]
            #         old_module.running_var.data[keep_mask] = new_module.running_var.data[keep_mask]
            #     else:
            #         old_module.running_mean.data = new_module.running_mean.data
            #         old_module.running_var.data = new_module.running_var.data

        # weight
        w_grad0 = new_module.weight.grad.clone().detach()
        if subnet_mask is not None:
            w_grad0.data[freeze_mask] = 0

        copy_param_grad(old_module.weight,w_grad0)
        if batch_idx%args.ps_batch == args.ps_batch-1:
            old_module.weight.grad = old_module.weight.grad_tmp.clone().detach()
            old_module.weight.grad_tmp = None

        # bias
        if hasattr(new_module,'bias') and new_module.bias is not None:
            b_grad0 = new_module.bias.grad.clone().detach()
            if subnet_mask is not None:
                b_grad0.data[freeze_mask] = 0

            copy_param_grad(old_module.bias,b_grad0)
            if batch_idx%args.ps_batch == args.ps_batch-1:
                old_module.bias.grad = old_module.bias.grad_tmp.clone().detach()
                old_module.bias.grad_tmp = None
            
    def copy_param_grad(old_param,new_grad):
        new_grad *= args.alphas[net_id]
        if not hasattr(old_param,'grad_tmp') or old_param.grad_tmp is None:
            old_param.grad_tmp = new_grad
        else:
            old_param.grad_tmp += new_grad

    bns1,convs1 = old_model.get_sparse_layers_and_convs()
    bns2,convs2 = new_model.get_sparse_layers_and_convs()
    ch_start = 0
    for conv1,bn1,conv2,bn2 in zip(convs1,bns1,convs2,bns2):
        ch_len = conv1.weight.data.size(0)
        with torch.no_grad():
            subnet_mask = mask[ch_start:ch_start+ch_len]
            copy_module_grad(bn1,bn2,subnet_mask)
            copy_module_grad(conv1,conv2,subnet_mask)
        ch_start += ch_len
    
    with torch.no_grad():
        old_non_sparse_modules = get_non_sparse_modules(old_model)
        new_non_sparse_modules = get_non_sparse_modules(new_model)
        for old_module,new_module in zip(old_non_sparse_modules,new_non_sparse_modules):
            copy_module_grad(old_module,new_module)


#baseline - original: def train(epoch, weight_valid_mask):
def train(epoch, weight_valid_mask=None):
    model.train()
    global history_score, global_step
    avg_loss = 0.
    avg_sparsity_loss = 0.
    train_acc = 0.
    total_data = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.noise_ratio == 0.0:
            pass
        else:
            mod = int(1/args.noise_ratio)
            #old random noise: target[0::mod].random_(0, 10)
            #symmetric noise:
            target[0::mod] = 9 - target[0::mod]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        nonzero = np.array([1, args.remain_ratio])
        #separation = int(nonzero[batch_idx%len(nonzero)])
        separation = 1
        if args.loss in {LossType.ONESHOT}:
            if separation == 0:
                remain_ratio = 1
            else:
                remain_ratio = args.remain_ratio
            weight_valid_mask = create_mask(model, remain_ratio)
            # =============================================== #
            #sample network: set "weights" in 'model' to 0
            #sample_network(model, weight_valid_mask)
            # =============================================== #
            freeze_mask, dynamic_model, ch_indices = sample_network_dynamic(model, weight_valid_mask, separation)
            if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
                output = dynamic_model(data)
        elif args.loss in {LossType.ITERATIVE}:
            sample_network(model, weight_valid_mask)
            output = model(data)
        else:
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
        if args.loss in {LossType.PROGRESSIVE_SHRINKING}:
            update_shared_model(model, dynamic_model, freeze_mask, batch_idx, ch_indices, separation)
        optimizer.step()
        if args.loss in {LossType.POLARIZATION,
                         LossType.L2_POLARIZATION}:
            clamp_bn(model, upper_bound=args.clamp)
        global_step += 1
        if batch_idx % args.log_interval == 0:
            print('Step: {} Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                global_step, epoch, batch_idx * len(data), len(train_loader.dataset),
                                    100. * batch_idx / len(train_loader), loss.data.item()))

    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = float(train_acc) / float(total_data)
    history_score[epoch][3] = avg_sparsity_loss / len(train_loader)


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    test_iter = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_iter):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            if isinstance(output, tuple):
                output, output_aux = output
            test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath, backup: bool, backup_path: str, epoch: int, max_backup: int):
    state['args'] = args

    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    if backup and backup_path is not None:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(backup_path, 'checkpoint_{}.pth.tar'.format(epoch)))

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

if args.flops_weighted:
    writer.add_text("train/conv_flops_weight", flops_weight_string, global_step=0)

for module_name, bn_module in model.named_modules():
    if not isinstance(bn_module, nn.BatchNorm2d) and not isinstance(bn_module, nn.BatchNorm1d):
        continue
    for i in range(2):
        bn_module.register_buffer(f"mean{i}",bn_module.running_mean.data.clone().detach())
        bn_module.register_buffer(f"var{i}",bn_module.running_var.data.clone().detach())

#weight_valid_mask = create_mask(model, args.remain_ratio)
if args.oneshot_ltr:
    model._initialize_weights()

# debugging save checkpoint
# is_best = True
# if not os.path.exists(args.save + 'test/'):
#     os.makedirs(args.save + 'test/')
# save_checkpoint({
#     'epoch': 0 + 1,
#     'state_dict': model.state_dict(),
#     'best_prec0': 0,
#     'optimizer': optimizer.state_dict(),
# }, is_best,
#     filepath= args.save + 'test/',
#     backup_path=args.backup_path,
#     backup=0 % args.backup_freq == 0,
#     epoch=0,
#     max_backup=args.max_backup
# )
# ckpt=torch.load(args.save + 'test/checkpoint.pth.tar')
# print(ckpt["epoch"])
# exit(0)
# debugging save checkpoint

if args.loss in {LossType.ITERATIVE}:
    print("here")
    remain_ratio = args.remain_ratio
    pruning_step = args.pruning_step
    weight_valid_mask = None
    while weight_valid_mask is None or weight_valid_mask.float().mean() > remain_ratio+1e-6:
        weight_valid_mask = iter_create_mask(model, weight_valid_mask, remain_ratio, pruning_step)
        model._initialize_weights()
        best_prec0 = 0
        for epoch in range(args.start_epoch, args.epochs):
            if args.max_epoch is not None and epoch >= args.max_epoch:
                break
            current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
            print("Start epoch {}/{} with learning rate {}...".format(epoch, args.epochs, current_learning_rate))
            weights, bias = bn_weights(model)
            train(epoch, weight_valid_mask)
            prec0 = test(model)
            is_best = prec0 > best_prec0
            best_prec0 = max(prec0, best_prec0)
            print(f"model prec :{prec0:.2f}")
            if not os.path.exists(args.save + str(weight_valid_mask.float().mean())):
                os.makedirs(args.save + str(weight_valid_mask.float().mean()))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec0': prec0,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath= args.save + str(weight_valid_mask.float().mean()),
                backup_path=args.backup_path,
                backup=epoch % args.backup_freq == 0,
                epoch=epoch,
                max_backup=args.max_backup
            )
            ##
            #ckpt=torch.load(args.save + '{:.1f}/checkpoint.pth.tar'.format(weight_valid_mask.float().mean()))
            #print(ckpt["epoch"])
            ##
            print("Epoch accuracy: " + str(best_prec0) + " " + str(prec0))
            with open(os.path.join(args.save + str(weight_valid_mask.float().mean()) + 'Prec'), 'a+') as fp:
                fp.write(f'{epoch} {best_prec0} {prec0}\n')

        print("Best accuracy: " + str(best_prec0))
        with open(os.path.join(args.save, 'Overall_Prec_Record.txt'), 'a+') as fp:
            fp.write(f'{weight_valid_mask.float().mean()} {best_prec0}\n')

else:
    for epoch in range(args.start_epoch, args.epochs):
        if args.max_epoch is not None and epoch >= args.max_epoch:
            break
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.decay_epoch)
        print("Start epoch {}/{} with learning rate {}...".format(epoch, args.epochs, current_learning_rate))
        weights, bias = bn_weights(model)
        train(epoch)
        #if args.prune_scale = 0:
        weight_valid_mask_original = create_mask(model, 1)
        weight_valid_mask_pruned = create_mask(model, args.remain_ratio)
        # 0: original model separation
        # 1: pruned model separation
        orginal_model = sample_network_dynamic(model, weight_valid_mask_original, 0, eval=True)
        dynamic_model = sample_network_dynamic(model, weight_valid_mask_pruned, 1, eval=True)
        #from resprune_gate import prune_resnet
        #pruned_model = prune_resnet(sparse_model=dynamic_model, pruning_strategy='fixed', prune_type='mask',
        #                    sanity_check=False, prune_mode="default", num_classes=num_classes, inplace_prune=True)
        prec0 = test(orginal_model)
        prec1 = test(dynamic_model)
        print(f"original model prec0 :{prec0:.2f}")
        print(f"pruned model prec1 :{prec1:.2f}")


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

    # flops
    # peek the remaining flops
    if args.loss in {LossType.POLARIZATION, LossType.L2_POLARIZATION}:
        flops_grad, flops_fixed, baseline_flops = prune_while_training(model, arch=args.arch,
                                                                       prune_mode="default",
                                                                       num_classes=num_classes)
        print(f" --> FLOPs in epoch (grad) {epoch}: {flops_grad:,}, ratio: {flops_grad / baseline_flops}")
        print(f" --> FLOPs in epoch (fixed) {epoch}: {flops_fixed:,}, ratio: {flops_fixed / baseline_flops}")

        if args.loss == LossType.POLARIZATION and args.target_flops and (
                flops_grad / baseline_flops) <= args.target_flops and args.gate:
            print("The grad pruning FLOPs archieve the target FLOPs.")
            print(f"Current pruning ratio: {flops_grad / baseline_flops}")
            print("Stop polarization from current epoch and continue training.")

            # do not apply polarization loss
            args.lbd = 0
            freeze_sparse_gate(model)
            if args.backup_freq > 20:
                args.backup_freq = 20

if args.loss == LossType.POLARIZATION and args.target_flops and (
        flops_grad / baseline_flops) > args.target_flops and args.gate:
    print("WARNING: the FLOPs does not achieve the target FLOPs at the end of training.")
print("Best accuracy: " + str(prec0))
history_score[-1][0] = prec0
np.savetxt(os.path.join(args.save, 'overall_record.txt'), history_score, fmt='%10.5f', delimiter=',')

#writer.close()

print("Best accuracy: " + str(prec0))