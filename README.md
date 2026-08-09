# IMM-Byte: Identity-Consistent Fish Tracking for Underwater Images
An enhanced version of ByteTrack for tracking fish in underwater low-light, occlusion based images. IMM-Byte replaces ByteTrack's single motion model with an Interacting Multiple Model (IMM) filter (Constant Velocity + Constant Acceleration) and introduces FishIoU, an association metric adapted for underwater occlusions, so that individual fish keep the same identity as they swim past camera obstructions, accelerate suddenly, or turn sharply.

<p align="center">
  <img src="Images/Full_function_diagram_v4.png" width="900" alt="Overview of the IMM-Byte tracking framework">
</p>

<p align="center">
  <em>Overview of the proposed IMM-Byte framework.</em>
</p>

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

# Installation
git clone https://github.com/<your-username>/IMM-Byte.git
cd IMM-Byte

conda create -n immbyte python=3.9
conda activate immbyte

pip install -r requirements.txt

# Tested environment
| Component | Version
| --- | --- | --- | --- |
| Python| - | 
| PyTorch | - | 
| CUDA | - |
|GPU | 	NVIDIA A100 (80GB)?? |

#Pretrained weights
<!-- FILL IN: this section is critical for reproducibility. For each weight file, specify: --> <!-- - what it is, what dataset it was trained on, where to download it, and its license -->
| Weight | Description | Trained on | Download | License
| --- | --- | --- | --- | --- |
| - | - | - | - | - |
| yolox_x_coco.pth | Original YOLOX-X COCO-pretrained backbone used as initialization | MS COCO | Official YOLOX release |-|

# To use the pretrained weights:
mkdir weights
# download yolox_x_gfisherd.pth into weights/
python tools/track.py --weights weights/yolox_x_gfisherd.pth --source <video_or_sequence>

# What we changed from baseline ByteTrack
This section is aimed at anyone extending or auditing the code — it summarizes every modification relative to the original ByteTrack repository.
# 1. Motion model: single Kalman Filter → IMM (CV + CA)
- Baseline: ByteTrack predicts each track's next position using a single Kalman filter with a constant-velocity (CV) motion assumption (tracker/kalman_filter.py, unmodified, kept for ablation comparisons).
- Our change: tracker/imm_filter.py (new file) implements an Interacting Multiple Model estimator that runs two parallel filters — CV and Constant Acceleration (CA) — and fuses their outputs each frame based on continuously updated mode probabilities (Blom & Bar-Shalom, 1988).
- Why: fish alternate between steady cruising and burst acceleration; a single CV filter lags badly during bursts, causing identity switches.
- Mode transition matrix used: Π = [[0.95, 0.05], [0.09, 0.91]] (CV↔CA) <!-- FILL IN if you tuned this differently -->

# 2. Association metric: standard IoU → FishIoU
- Baseline: ByteTrack associates detections to tracks using bounding-box IoU.
- Our change: tracker/fish_iou.py (new file) implements a modified association cost:
FishIoU = ω1·IoU + ω2·aspect_ratio_consistency + ω3·area_consistency − ω4·scaled_center_distance
with ω1=1, ω2=0.2, ω3=0.2, ω4=0.6 (empirically tuned for this dataset).

Note: we deliberately omit the central-region IoU (cIoU) term from Li et al. (2024)'s original FishIoU formulation, since our camera setup has vertical bar occlusions that corrupt center-region overlap and cause false identity switches (see paper Section III-D / Fig. 12).
# 3. Two-stage association (unchanged from ByteTrack)
We retain ByteTrack's high-confidence / low-confidence two-stage matching strategy (tracker/byte_tracker.py), simply substituting FishIoU as the cost function in place of standard IoU, and substituting IMM-predicted states in place of single-KF-predicted states.

# Usage
# Run tracking on a video/sequence
bash
python tools/track.py \
  --weights weights/yolox_x_gfisherd.pth \
  --source path/to/video_or_sequence \
  --conf-thresh 0.5 \
  --output results/

# Evaluate tracking metrics (IDF1, MOTA, IDSW, etc.)
bash
python tools/eval_motmetrics.py \
  --gt path/to/ground_truth \
  --pred results/ \
  --output eval_report.txt
GFISHERD24 — comparison with state-of-the-art trackers
Method	HOTA↑	DetA↑	AssA↑	IDF1↑	MOTA↑	IDSW↓
OC-SORT	39.44	32.72	48.30	41.49	<!-- FILL IN: verify against appendix -->	281
BoT-SORT	46.25	41.65	52.32	52.20	57.50%	439
ByteTrack	44.32	42.04	47.34	62.28	60.90%	485
ByteTrack + IMM	45.34	41.96	49.53	64.80	60.78%	318
IMM-Byte (proposed)	45.08	42.26	48.67	63.70	61.30%	253

Full per-sequence breakdowns are provided in the paper's Appendix and in results/per_sequence/.

DanceTrack — generalization
Method	HOTA	DetA	AssA	IDF1	MOTA	IDSW↓
ByteTrack (baseline)	46.00	70.34	30.21	51.38	88.29%	1987
ByteTrack + IMM	46.23	70.60	30.40	50.95	88.35%	1821

Citation

If you use this code or the GFISHERD24 dataset, please cite:

<!-- FILL IN: final BibTeX once the paper is published/has a DOI -->
bibtex
@article{ebu2026immbyte,
  title   = {Identity-Consistent Multi-Object Tracking via Interacting Multiple
             Model Kalman Filtering for Fish Species Monitoring},
  author  = {Ebu, Iffat Ara and Nabi, M M and Moorhead, Robert},
  journal = {<!-- FILL IN -->},
  year    = {2026}
}
Acknowledgments
Built on top of ByteTrack (Zhang et al., 2022)
Detector based on YOLOX (Ge et al., 2021)
FishIoU formulation adapted from Li et al. (2024), "When trackers date fish"
GFISHERD24 data collected by NOAA as part of its annual fishery-independent reef fish survey
<!-- FILL IN: funding sources, lab affiliation, etc. if applicable -->
Contact
<!-- FILL IN: your email or lab page, for marine biologists who may want to reach out about using the tool on their own survey footage -->

For questions about using this tool on your own survey data, open an issue or contact <!-- FILL IN -->.
