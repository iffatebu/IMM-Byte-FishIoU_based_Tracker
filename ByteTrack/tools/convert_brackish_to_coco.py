"""
Convert BrackishMOT (train/ and test/, each with <seq>/img1, <seq>/gt/gt.txt,
<seq>/seqinfo.ini) into YOLOX/ByteTrack's expected COCO-with-video-fields
JSON format.

Output: train.json and test.json, each containing:
    images:     [{id, file_name, frame_id, prev_image_id, next_image_id,
                  video_id, height, width}, ...]
    annotations:[{id, category_id, image_id, track_id, bbox, area, iscrowd,
                  conf}, ...]
    videos:     [{id, file_name}, ...]
    categories: [{id, name}, ...]   # 1-6, matching BrackishMOT's gt.txt classes

file_name is written as "<split>/<seq>/img1/<frame>.jpg" (e.g.
"train/BrackishMOT-01/img1/000001.jpg") so you can point the Exp file's
data_dir straight at the BrackishMOT root and everything resolves correctly,
whether train and test sequences live under separate top-level folders or not.

Usage:
    python convert_brackish_to_coco.py --root /scratch/morrill/users/ie93/ByteTrack/datasets/BrackishMOT --split train --out /scratch/morrill/users/ie93/ByteTrack/datasets/BrackishMOT/annotations/train.json
    python convert_brackish_to_coco.py --root /scratch/morrill/users/ie93/ByteTrack/datasets/BrackishMOT --split test  --out /scratch/morrill/users/ie93/ByteTrack/datasets/BrackishMOT/annotations/test.json
"""

import argparse
import configparser
import csv
import json
from pathlib import Path
from PIL import Image

CATEGORIES = [
    {"id": 1, "name": "fish"},
    {"id": 2, "name": "crab"},
    {"id": 3, "name": "shrimp"},
    {"id": 4, "name": "starfish"},
    {"id": 5, "name": "small_fish"},
    {"id": 6, "name": "jellyfish"},
]


def find_images_dir(seq_path: Path):
    for cand in ["img1", "images", "img"]:
        p = seq_path / cand
        if p.exists():
            return p
    return seq_path


def read_seqinfo(seq_path: Path):
    ini_path = seq_path / "seqinfo.ini"
    width = height = None
    if ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        if "Sequence" in cfg:
            width = cfg["Sequence"].getint("imWidth", fallback=None)
            height = cfg["Sequence"].getint("imHeight", fallback=None)
    return width, height


def convert(root: Path, split: str, out_path: Path):
    split_root = root / split
    seq_dirs = sorted([d for d in split_root.iterdir() if d.is_dir()])
    if not seq_dirs:
        raise RuntimeError(f"No sequence subfolders found under {split_root}")

    out = {"images": [], "annotations": [], "videos": [], "categories": CATEGORIES}
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0

    for seq_path in seq_dirs:
        seq_name = seq_path.name
        video_cnt += 1
        out["videos"].append({"id": video_cnt, "file_name": seq_name})

        img_dir = find_images_dir(seq_path)
        width, height = read_seqinfo(seq_path)

        image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not image_files:
            print(f"  [warn] no images found for {seq_name}")
            continue

        frame_to_imgid = {}
        num_images = len(image_files)
        base_image_cnt = image_cnt

        for i, f in enumerate(image_files):
            frame_num = int("".join(ch for ch in f.stem if ch.isdigit()))
            w, h = width, height
            if w is None or h is None:
                with Image.open(f) as im:
                    w, h = im.size

            img_id = base_image_cnt + i + 1
            frame_to_imgid[frame_num] = img_id
            out["images"].append({
                "id": img_id,
                "file_name": f"{split}/{seq_name}/{img_dir.name}/{f.name}",
                "frame_id": i + 1,
                "prev_image_id": img_id - 1 if i > 0 else -1,
                "next_image_id": img_id + 1 if i < num_images - 1 else -1,
                "video_id": video_cnt,
                "height": h,
                "width": w,
            })
        image_cnt += num_images

        gt_path = seq_path / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"  [warn] no gt.txt for {seq_name} (ok if this is an unlabeled test split)")
            continue

        n_ann_this_seq = 0
        with open(gt_path, newline="") as gf:
            for row in csv.reader(gf):
                if not row:
                    continue
                frame, obj_id, left, top, w, h, conf, cls, vis = row[:9]
                conf = float(conf)
                if conf == 0:
                    continue
                frame = int(frame)
                if frame not in frame_to_imgid:
                    continue
                left, top, w, h = float(left), float(top), float(w), float(h)
                ann_cnt += 1
                n_ann_this_seq += 1
                out["annotations"].append({
                    "id": ann_cnt,
                    "category_id": int(float(cls)),  # already 1-6, matches CATEGORIES ids
                    "image_id": frame_to_imgid[frame],
                    "track_id": int(obj_id),
                    "bbox": [left, top, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "conf": conf,
                })

        print(f"  [ok] {seq_name}: {num_images} images, {n_ann_this_seq} boxes")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {len(out['images'])} images, {len(out['annotations'])} annotations, "
          f"{len(out['videos'])} videos to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to BrackishMOT root (contains train/ and test/)")
    ap.add_argument("--split", type=str, choices=["train", "test"], required=True)
    ap.add_argument("--out", type=str, required=True, help="Output json path")
    args = ap.parse_args()

    convert(Path(args.root), args.split, Path(args.out))