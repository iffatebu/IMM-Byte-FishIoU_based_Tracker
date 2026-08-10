#!/bin/bash

#SBATCH --job-name=training

#SBATCH --partition=gpu-a100

#SBATCH --account=partner-ngi

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:1

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G

#SBATCH --time=32:00:00

#SBATCH --output=sbatch_output/tracking_performance_output_%j.out

#SBATCH --error=sbatch_output/tracking_performance_error_%j.out

# --- env sanity ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- NCCL stability ---
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

#Running the model for tracking

python3 tools/Tracking_Performance.py image -f exps/example/mot/yolox_nano_brackish.py -c YOLOX_outputs/yolox_nano_brackish/latest_ckpt.pth.tar --fp16 --fuse --save_result

