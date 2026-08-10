"""
Draw BOTH ground truth boxes (red) and model detections (green/labeled) on the
same frame, so you can see directly whether detections are anywhere near real
fish, or just noise somewhere else in the image.

Usage:
    python visualize_gt_vs_detections.py \
        -f exps/example/mot/yolox_nano_brackish.py \
        -c YOLOX_outputs/yolox_nano_brackish/best_ckpt.pth.tar \
        --path datasets/BrackishMOT/test/brackishMOT-93/img1/000123.jpg \
        --gt_txt datasets/BrackishMOT/test/brackishMOT-93/gt/gt.txt \
        --class_id 5 \
        --conf 0.1 --nms 0.85 \
        --save_path vis_gt_vs_pred.jpg
"""

import argparse
import csv

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

CLASS_NAMES = ("fish", "crab", "shrimp", "starfish", "small_fish", "jellyfish")


def load_gt_boxes(gt_path, frame_num, class_id=None):
    boxes = []
    with open(gt_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            frame, obj_id, left, top, w, h, conf, cls, vis_ = row[:9]
            if int(frame) != frame_num:
                continue
            if float(conf) == 0:
                continue
            if class_id is not None and int(float(cls)) != class_id:
                continue
            boxes.append((float(left), float(top), float(w), float(h), obj_id, int(float(cls))))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--exp_file", required=True)
    ap.add_argument("-c", "--ckpt", required=True)
    ap.add_argument("--path", required=True, help="path to a single image, e.g. .../img1/000123.jpg")
    ap.add_argument("--gt_txt", required=True, help="path to the matching gt.txt for this sequence")
    ap.add_argument("--class_id", type=int, default=None,
                     help="restrict GT overlay to one class, e.g. 5 for small_fish. Omit for all classes.")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--nms", type=float, default=0.85)
    ap.add_argument("--save_path", type=str, default="vis_gt_vs_pred.jpg")
    args = ap.parse_args()

    frame_num = int(args.path.split("/")[-1].split(".")[0])

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
    result_img = img.copy()
    n_dets = 0
    if output is not None:
        output = output.cpu()
        bboxes = output[:, 0:4]
        scale = min(exp.test_size[0] / orig_h, exp.test_size[1] / orig_w)
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        result_img = vis(img, bboxes.numpy(), scores.numpy(), cls.numpy(), args.conf, CLASS_NAMES)
        n_dets = len(bboxes)

    gt_boxes = load_gt_boxes(args.gt_txt, frame_num, args.class_id)
    for (left, top, w, h, obj_id, cls_id) in gt_boxes:
        x1, y1, x2, y2 = int(left), int(top), int(left + w), int(top + h)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = ground truth
        cv2.putText(result_img, f"GT#{obj_id}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(args.save_path, result_img)
    print(f"Frame {frame_num}: {n_dets} model detections (green), {len(gt_boxes)} ground truth boxes (red)")
    print(f"Saved overlay to {args.save_path}")


if __name__ == "__main__":
    main()