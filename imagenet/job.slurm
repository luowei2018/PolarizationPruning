#!/bin/bash
#SBATCH --job-name="bo_test"
#SBATCH --output="bo_test.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=220G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbng-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 10:00:00
cd /scratch/bbng/mw34/bo/PolarizationPruning/imagenet/
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate boenv
export OMP_NUM_THREADS=8  # if code is not multithreaded, otherwise set to 8 or 16
srun -N 1 -n 4 python3 -W ignore -u main_oneshot.py /scratch/bbng/mw34/datasets/imagenet -loss ps -b 512 --lbd 8e-5 --t 1.2 --lr-strategy step --lr 1e-3 1e-4 --decay-epoch 80 --epochs 160 --arch resnet50 --workers 8 --world-size 1 --dist-url tcp://localhost:23456 --dist-backend gloo --rank 0 --save ./ps23_05_1 --resume ps23_07_1/checkpoint.pth.tar --load-param-only --alphas 0 0 0.5 1 --split_running_stat --load_running_stat > ./ps23_05_1/myjob.out