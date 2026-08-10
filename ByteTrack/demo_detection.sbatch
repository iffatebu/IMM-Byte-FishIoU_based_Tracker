#!/bin/bash

#SBATCH --job-name=training

#SBATCH --partition=gpu-a100

#SBATCH --account=partner-ngi

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:4

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G

#SBATCH --time=32:00:00

#SBATCH --output=sbatch_output/byteTrack_output_%j.out

#SBATCH --error=sbatch_output/byteTrack_error_%j.out

# --- env sanity --- 
export CUDA_DEVICE_ORDER=PCI_BUS_ID 
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# --- NCCL stability --- 
export NCCL_DEBUG=INFO 
export NCCL_ASYNC_ERROR_HANDLING=1 
export NCCL_IB_DISABLE=1 
export NCCL_SOCKET_IFNAME=^lo,docker0 

echo "Host: $(hostname)" 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 
nvidia-smi 

#python3 tools/demo_detection_fish.py -f /scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_x_ablation/100Train_16Test/testConf_0.01_nms_0.5/YOLOX_X/yolox_x_ablation.py -c pretrained/best_ckpt.pth.tar --fp16 --fuse --save_result True


python tools/visualize_gt_vs_detections.py \
    -f exps/example/mot/yolox_nano_brackish.py \
    -c YOLOX_outputs/yolox_nano_brackish/best_ckpt.pth.tar \
    --path datasets/BrackishMOT/test/brackishMOT-93/img1/000128.jpg \
    --gt_txt datasets/BrackishMOT/test/brackishMOT-93/gt/gt.txt \
    --class_id 5 \
    --conf 0.1 --nms 0.85 \
    --save_path vis_gt_vs_pred.jpg