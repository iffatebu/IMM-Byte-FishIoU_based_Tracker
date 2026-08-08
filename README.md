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

## Repository Structure

```text
IMM-Byte/
│
├── detector/
│   └── yolox_x_gfisherd.py
│       # YOLOX-X detector configuration for GFISHERD24
│
├── tracker/
│   ├── byte_tracker.py
│   │   # Modified ByteTrack tracking and association logic
│   │
│   ├── imm_filter.py
│   │   # Interacting Multiple Model (IMM) estimator
│   │   # using Constant Velocity (CV) and Constant Acceleration (CA) models
│   │
│   ├── fish_iou.py
│   │   # FishIoU association metric for underwater fish tracking
│   │
│   └── kalman_filter.py
│       # Original single-model Kalman Filter
│       # retained for baseline comparison and ablation studies
│
├── configs/
│   └── gfisherd24.yaml
│       # Dataset, detector, and tracking hyperparameters
│
├── weights/
│   # Directory for pretrained model weights
│
├── tools/
│   ├── train.py
│   │   # Detector training script
│   │
│   ├── track.py
│   │   # Tracking inference on videos or image sequences
│   │
│   └── eval_motmetrics.py
│       # MOT evaluation for IDF1, MOTA, IDSW, and other metrics
│
├── datasets/
│   └── README.md
│       # Dataset preparation and setup instructions
│       # for GFISHERD24 and DanceTrack
│
├── requirements.txt
│   # Python dependencies
│
└── README.md
    # Project documentation
```

### Directory and File Descriptions

| Path                           | Description                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `detector/`                    | Contains the YOLOX-X object detection configuration and implementation used for fish detection.                     |
| `detector/yolox_x_gfisherd.py` | YOLOX-X detector configured for the GFISHERD24 dataset.                                                             |
| `tracker/`                     | Contains the core multi-object tracking, motion estimation, and association components.                             |
| `tracker/byte_tracker.py`      | Modified ByteTrack implementation containing the primary tracking and data-association logic.                       |
| `tracker/imm_filter.py`        | Implements the IMM estimator with parallel CV and CA Kalman filters for adaptive motion prediction.                 |
| `tracker/fish_iou.py`          | Implements the FishIoU association metric used to improve spatial matching between detections and predicted tracks. |
| `tracker/kalman_filter.py`     | Original single-model Kalman Filter retained for baseline comparison and ablation experiments.                      |
| `configs/gfisherd24.yaml`      | Contains dataset paths, detector settings, tracking parameters, and other experiment-specific hyperparameters.      |
| `weights/`                     | Stores pretrained YOLOX-X model checkpoints and other required model weights.                                       |
| `tools/train.py`               | Script for training the object detector.                                                                            |
| `tools/track.py`               | Inference script for running IMM-Byte on videos or image sequences.                                                 |
| `tools/eval_motmetrics.py`     | Evaluation script for calculating standard MOT metrics, including IDF1, MOTA, and IDSW.                             |
| `datasets/README.md`           | Provides instructions for downloading, organizing, and preparing GFISHERD24 and DanceTrack datasets.                |
| `requirements.txt`             | Lists the Python dependencies required to run the project.                                                          |
| `README.md`                    | Main documentation for installation, usage, methodology, experiments, and reproduction.                             |



