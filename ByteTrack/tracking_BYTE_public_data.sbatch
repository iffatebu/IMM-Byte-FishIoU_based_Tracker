#!/bin/bash

#SBATCH --job-name=tracking

#SBATCH --partition=gpu-a100

#SBATCH --account=partner-ngi

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:1                   

#SBATCH --cpus-per-task=16

#SBATCH --mem=64G

#SBATCH --time=32:00:00

#SBATCH --output=sbatch_output/tracking_output_%j.out

#SBATCH --error=sbatch_output/tracking_error_%j.out

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

#Running the model for Regualr Bytetrack tracking
#python3 tools/track.py images -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
# Dancetrack validation
#python3 tools/track.py -f exps/example/dancetrack/yolox_x.py -c YOLOX_outputs/yolox_x/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse

# BrackishMOT multiclass test
# python3 tools/track_multiclass.py -f exps/example/mot/yolox_nano_brackish.py -c YOLOX_outputs/yolox_nano_brackish/best_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse --test

# BrackishMOT single-class test
#python3 tools/track.py -f exps/example/mot/yolox_x_ablation_brackish.py -c YOLOX_outputs/yolox_x_ablation_brackish/latest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse --test
python3 tools/demo_updated_brackish_singleClassTrack.py images -f /scratch/morrill/users/ie93/ByteTrack/exps/example/mot/yolox_x_ablation_brackish.py -c /scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_x_ablation_brackish/latest_ckpt.pth.tar --fp16 --fuse --save_result



# python3 tools/track_multiclass.py -f exps/example/mot/yolox_nano_brackish.py \
#     -c YOLOX_outputs/yolox_nano_brackish/best_ckpt.pth.tar \
#     -b 1 -d 1 --fp16 --fuse --test
#     --track_thresh 0.3 --conf 0.03 --nms 0.85


# for i in 0 1 2 3 4 5 6 7; do                      #>>>>>>Change here: For GPU number
#     CUDA_VISIBLE_DEVICES=$i python3 tools/demo_updated_all_singleClassTrack.py images \
#         -f /scratch/morrill/users/ie93/ByteTrack_old/exps/example/mot/Iffat_model_For410Train/yolox_x_ablation.py \
#         -c pretrained/best_ckpt.pth.tar \
#         --fp16 --fuse --save_result \
#         --gpu_id $i \
#         --total_gpus 8 \                          #>>>>>>Change here: For GPU number
#         --gpu_rank $i &
# done
# wait

# for i in 0 1 2 3; do                                
#     CUDA_VISIBLE_DEVICES=$i python3 tools/track.py \
#         -f exps/example/mot/yolox_x_ablation.py \
#         -c pretrained/bytetrack_ablation.pth.tar \
#         --fp16 --fuse --save_result \
#         --gpu_id $i \
#         --total_gpus 4 \
#         --gpu_rank $i &
# done
# wait

# For single Gpu use
# Single GPU run — no loop needed
# CUDA_VISIBLE_DEVICES=0 python3 tools/demo_updated_all_singleClassTrack.py images \
#     -f /scratch/morrill/users/ie93/ByteTrack_old/exps/example/mot/Iffat_model_For410Train/yolox_x_ablation.py \
#     -c pretrained/best_ckpt.pth.tar \
#     --fp16 --fuse --save_result \
#     --gpu_id 0 \
#     --total_gpus 1 \
#     --gpu_rank 0

#Brackish per class evaluation
# python eval_per_class_ap.py \
#     -f exps/example/mot/yolox_nano_brackish.py \
#     -c YOLOX_outputs/yolox_nano_brackish/latest_ckpt.pth.tar \
#     -b 8 --ann_file test.json
