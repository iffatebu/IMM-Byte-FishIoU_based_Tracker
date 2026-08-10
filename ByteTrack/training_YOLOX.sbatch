#!/bin/bash

#SBATCH --job-name=training

#SBATCH --partition=gpu-a100

#SBATCH --account=partner-ngi

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:8

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G

#SBATCH --time=48:00:00

#SBATCH --output=sbatch_output/brackishMOT_YOLOX_output_%j.out

#SBATCH --error=sbatch_output/brackishMOT_YOLOX_error_%j.out

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

#Running the model for training

#python3 tools/train.py -f /scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_x_ablation/GFISHER_Whole_testConf_0.01_nms_0.5/YOLOX_X/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth

python3 tools/train.py -f exps/example/mot/yolox_x_ablation_brackish.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth

#Resume from previous weights
# python3 tools/train.py -f exps/example/mot/yolox_nano_brackish.py \
#     -d 6 -b 36 --fp16 -o --resume

