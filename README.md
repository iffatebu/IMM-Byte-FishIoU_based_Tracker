# IMM-Byte: Identity-Consistent Fish Tracking for Underwater Images
An enhanced version of ByteTrack for tracking fish in underwater low-light, occlusion based images. IMM-Byte replaces ByteTrack's single motion model with an Interacting Multiple Model (IMM) filter (Constant Velocity + Constant Acceleration) and introduces FishIoU, an association metric adapted for underwater occlusions, so that individual fish keep the same identity as they swim past camera obstructions, accelerate suddenly, or turn sharply.

Underwater survey cameras record fish swimming past structural bars, vegetation, and each other. Standard tracking software often "loses" a fish mid-video and assigns it a new identity when it reappears which is called an identity switch (IDSW). If a fish is double-counted this way, abundance estimates for fisheries management become unreliable. IMM-Byte reduces this problem substantially: compared to the widely used ByteTrack tracker, it cuts identity switches by 47.84% (from 485 to 253) on our GFISHERD24 survey dataset, while also improving overall tracking accuracy.


| Metric | ByteTrack<br>(baseline)| IMM-Byte<br>(proposed) | IMM-Byte-FishIoU<br>(proposed)
| --- | --- | --- | --- |
| Identity Switches (IDSW)↓| 485 | 318 | 253 |
| MOTA ↑ | 60.6% | 60.6%  | 61.10%  |
| IDF1 ↑ | 	62.70 | 64.80 | 63.60 |

## Table of Contents
- Repository Structure
- Installation
- Pretrained weight (optional)
- Poposal
- Usage
- Datasets (optional)
- Results
- Citation
- Acknowledgments

IMM-Byte/
├── detector/                 # YOLOX-X detection code and configurations
│   └── yolox_x_gfisherd.py   # YOLOX-X detector configured for GFISHERD24
├── tracker/                  # Core tracking and association algorithms
│   ├── byte_tracker.py       # Modified ByteTrack association logic
│   ├── imm_filter.py         # NEW: IMM estimator (CV + CA Kalman filters)
│   ├── fish_iou.py           # NEW: FishIoU association metric
│   └── kalman_filter.py      # Original single-model KF (for comparison/ablation)
├── configs/
│   └── gfisherd24.yaml       # Dataset and tracking hyperparameters
├── weights/                  # Directory for pretrained model weights
├── tools/
│   ├── train.py              # Detector training script
│   ├── track.py              # Inference script to run tracking on video/sequences
│   └── eval_motmetrics.py    # Evaluation script for IDF1, MOTA, IDSW, etc.
├── datasets/
│   └── README.md             # Dataset setup instructions (GFISHERD24 / DanceTrack)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation



