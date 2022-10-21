#!/bin/bash
#SBATCH --job-name="m2"
#SBATCH --output="m2_s0_c01.5v2.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbod-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00
cd /scratch/bbng/boc2/PolarizationPruning/imagenet/
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate boenv
export OMP_NUM_THREADS=16  # if code is not multithreaded, otherwise set to 16 or 16

# baselines
# srun python3 -W ignore -u main.py /scratch/bbng/mw34/datasets/imagenet -loss sr -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./sr/resnet --resume ./sr/resnet/checkpoint.pth.tar
# srun python3 -W ignore -u main.py /scratch/bbng/mw34/datasets/imagenet -loss sr --warmup -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.4 --epochs 256 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./sr/mobilenetv2
# srun python3 -W ignore -u main.py /scratch/bbng/mw34/datasets/imagenet -loss zol -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./zol/resnet
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss zol --warmup -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.4 --epochs 256 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./zol/mobilenetv2 --resume ./zol/mobilenetv2/checkpoint.pth.tar.tmp
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss original -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./original2/resnet/ --resume ./original/resnet/model_best.pth.tar
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/boc2/datasets/imagenet -loss original --warmup -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.4 --epochs 256 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./original2/mobilenetv2 --start-epoch 128 --resume ./original/mobilenetv2/model_best.pth.tar 

# OFA baseline
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./ofa/resnet/ --resume original/resnet/model_best.pth.tar --load-param-only --OFA
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./ofav2/resnet/ --resume original/resnet/model_best.pth.tar --load-param-only --OFA
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps --warmup -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.4 --epochs 256 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./ofa/mobilenetv2/ --resume original/mobilenetv2/model_best.pth.tar --load-param-only --OFA

# OFA debug
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-1 1e-2 1e-3 --decay-epoch 40 80 --epochs 120 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./ofa/resnet/ --resume ./ofa/resnet/checkpoint.pth.tar --load-param-only --OFA


# my test
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps --warmup -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.04 --epochs 64 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./op/mobilenetv2/step0/ --resume op/mobilenetv2/step0/checkpoint.pth.tar  --alphas 0 0 1 1 --split_running_stat --load_running_stat
srun python3 -W ignore -u main_oneshot.py /scratch/bbng/boc2/datasets/imagenet -loss ps -b 1024 --lbd 2.5e-5 --t 1 --lr-strategy cos --lr 0.1 --epochs 32 --wd 0.00004 --no-bn-wd --arch mobilenetv2 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./op/mobilenetv2/step0/cos01v2/ --resume ./op/mobilenetv2/step0/cos01v2/checkpoint.pth.tar --ps_batch 8  --alphas 0 0 1 1 --split_running_stat --load_running_stat
# srun python3 -W ignore -u main_oneshot.py /scratch/bbng/boc2/datasets/imagenet -loss ps -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-2 1e-3 --decay-epoch 140 --epochs 160 --arch resnet50 --workers 16 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./op/resnet/step1/03/ --resume ./op/resnet/step1/03/checkpoint.pth.tar --ps_batch 12 --alphas 0 .3 1 .1 --split_running_stat --load_running_stat