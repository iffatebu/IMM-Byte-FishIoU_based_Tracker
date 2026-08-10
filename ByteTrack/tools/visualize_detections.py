"""
Standalone replacement for tools/demo.py (not present in this ByteTrack fork).
Runs raw detection on a single image and saves a visualization -- this
completely bypasses ByteTracker, so it shows you exactly what the detector
itself proposes, before any confidence/NMS/track-initiation logic touches it.

Usage:
    python visualize_detections.py \
        -f exps/example/mot/yolox_nano_brackish.py \
        -c YOLOX_outputs/yolox_nano_brackish/best_ckpt.pth.tar \
        --path datasets/BrackishMOT/test/brackishMOT-93/img1/000050.jpg \
        --conf 0.001 --nms 0.85 \
        --save_path vis_output.jpg
"""

import argparse

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

CLASS_NAMES = ("fish", "crab", "shrimp", "starfish", "small_fish", "jellyfish")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp_file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--path", required=True, help="path to a single image")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--save_path", type=str, default="vis_output.jpg")
    args = ap.parse_args()

    exp = get_exp(args.exp_file, None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    print("Building model and loading checkpoint...")
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    img = cv2.imread(args.path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {args.path}")
    orig_h, orig_w = img.shape[:2]

    preproc = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img_t, _ = preproc(img, None, exp.test_size)
    img_t = torch.from_numpy(img_t).unsqueeze(0).float().cuda()

    with torch.no_grad():
        outputs = model(img_t)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

    output = outputs[0]
    if output is None:
        print(f"No detections at all, even at conf={args.conf}. "
              f"Saving the original image unmodified so you can inspect it directly.")
        cv2.imwrite(args.save_path, img)
        return

    output = output.cpu()
    bboxes = output[:, 0:4]
    scale = min(exp.test_size[0] / orig_h, exp.test_size[1] / orig_w)
    bboxes /= scale  # back to original image pixel coordinates

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    result_img = vis(img, bboxes.numpy(), scores.numpy(), cls.numpy(), args.conf, CLASS_NAMES)
    cv2.imwrite(args.save_path, result_img)
    print(f"Saved {len(bboxes)} detections to {args.save_path}")

    # quick per-class count, useful for the small_fish recall question specifically
    from collections import Counter
    counts = Counter(CLASS_NAMES[int(c)] for c in cls.numpy())
    print("Detections by class:", dict(counts))


if __name__ == "__main__":
    main()