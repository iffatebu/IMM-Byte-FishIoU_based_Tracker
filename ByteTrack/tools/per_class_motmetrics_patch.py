"""
Insert this in place of the direct `compare_dataframes(gt, ts)` call in your
track.py-analog script. It filters both ground truth and predictions down to
one class at a time before comparing, then reports per-class metrics and an
average across classes -- avoiding the class-blind matching that
mm.io.loadtxt(fmt='mot15-2D') otherwise does (it has no concept of a class
column at all).

Assumes gt.txt and your results .txt both have class in column index 7
(0-indexed), matching write_results_multiclass's format:
    frame,id,x1,y1,w,h,score,cls,-1,-1
"""

import csv
import tempfile
from pathlib import Path

CLASS_NAMES = {1: "fish", 2: "crab", 3: "shrimp", 4: "starfish", 5: "small_fish", 6: "jellyfish"}


def filter_rows_by_class(src_path, class_id):
    """Return a list of comma-joined rows from src_path where column 8 == class_id."""
    kept = []
    with open(src_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if int(float(row[7])) != class_id:
                continue
            kept.append(",".join(row))
    return kept


def evaluate_per_class(gtfiles, tsfiles, mm, mh):
    """
    gtfiles, tsfiles: lists of file paths (as already built in your script).
    mm, mh: the motmetrics module and metrics host you already have in scope.
    Returns: dict class_id -> summary DataFrame, plus prints an averaged table.
    """
    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    per_class_summaries = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for class_id, class_name in CLASS_NAMES.items():
            gt = {}
            ts = {}

            for f in gtfiles:
                seq_name = Path(f).parts[-3]
                rows = filter_rows_by_class(f, class_id)
                if not rows:
                    continue
                tmp_gt = tmp / f"gt_{class_id}_{seq_name}.txt"
                tmp_gt.write_text("\n".join(rows))
                gt[seq_name] = mm.io.loadtxt(str(tmp_gt), fmt="mot15-2D", min_confidence=1)

            for f in tsfiles:
                seq_name = Path(f).stem
                rows = filter_rows_by_class(f, class_id)
                tmp_ts = tmp / f"ts_{class_id}_{seq_name}.txt"
                tmp_ts.write_text("\n".join(rows))  # empty file is fine -- means no predictions for this class/seq
                ts[seq_name] = mm.io.loadtxt(str(tmp_ts), fmt="mot15-2D", min_confidence=-1)

            if not gt:
                print(f"[{class_name}] no ground truth rows found across all sequences, skipping")
                continue

            accs = []
            names = []
            for k, tsacc in ts.items():
                if k in gt:
                    accs.append(mm.utils.compare_to_groundtruth(gt[k], tsacc, "iou", distth=0.5))
                    names.append(k)

            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
            per_class_summaries[class_name] = summary
            print(f"\n=== {class_name} (class {class_id}) ===")
            print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    # Simple average across classes for the key metrics (standard for multi-class MOT reporting)
    import pandas as pd
    overall_rows = {name: s.loc["OVERALL"] for name, s in per_class_summaries.items()}
    avg_df = pd.DataFrame(overall_rows).T
    print("\n=== Per-class OVERALL rows, and mean across classes ===")
    print(avg_df)
    print("\nMean across classes:")
    print(avg_df.mean(numeric_only=True))

    return per_class_summaries