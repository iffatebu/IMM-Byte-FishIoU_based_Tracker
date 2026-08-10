# -*- coding: utf-8 -*-
"""
Changed required:
GT_ROOT = directory
RES_ROOT = directory

@author: ie93
"""

# This Python file uses the following encoding: utf-8

import glob, os, logging
from pathlib import Path, PurePosixPath        # pathlib is easier
import motmetrics as mm
import random
import warnings
from collections import OrderedDict

#Collect every pair that really exists

log = logging.getLogger()                        # use std-lib logger
mm.lap.default_solver = 'lap'                    # same as before

# -----------------------------------------------------------------------
# GT_ROOT  = Path('/scratch/morrill/users/ie93/ByteTrack_old/datasets/mot/test/Sample_9')
GT_ROOT  = Path('/scratch/morrill/users/ie93/ByteTrack/datasets/BrackishMOT/test') # @@@@@ Change: .txt files of GT @@@@@@
#GT_ROOT  = Path("/scratch/morrill/users/ie93/ByteTrack_old/datasets/mot/test")
# RES_ROOT = Path('/scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_x_ablation/Original_singleClss_410TrainVideo/results') # @@@@@ Change: .txt files of output @@@@@@
RES_ROOT = Path('/scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_nano_brackish/track_results')
# -----------------------------------------------------------------------

# test/*/gt/gt.txt
gt_files = {p.parents[1].name : p                       # key = sequence name
            for p in GT_ROOT.glob('*/*/gt.txt')}

# every *.txt the tracker wrote
res_files = {p.stem : p                                 # key = sequence name
             for p in RES_ROOT.glob('*.txt')}

common = sorted(set(gt_files) & set(res_files))         # ? intersection
missing_gt  = sorted(set(res_files) - set(gt_files))
missing_res = sorted(set(gt_files)  - set(res_files))

if missing_gt:
    log.warning("No ground-truth for: %s", ', '.join(missing_gt))
if missing_res:
    log.warning("No result file  for: %s", ', '.join(missing_res))
if not common:
    raise SystemExit("Nothing to compare!")

# -----------------------------------------------------------------------
# load all dataframes ----------------------------------------------------
gts = {name: mm.io.loadtxt(str(gt_files[name]), fmt='mot15-2D',
                           min_confidence=1)
       for name in common}

tss = {name: mm.io.loadtxt(str(res_files[name]), fmt='mot15-2D',
                           min_confidence=-1)
       for name in common}

#Compare & compute metrics once

# pairwise comparison ? list of Accumulators
accs   = []
names  = []
for name in common:
    log.info("Comparing %s", name)
    accs.append(mm.utils.compare_to_groundtruth(gts[name],
                                                tss[name],
                                                'iou', distth=0.7))
    names.append(name)

mh = mm.metrics.create()

metrics = ['recall', 'precision',
           'num_unique_objects', 'mostly_tracked', 'partially_tracked',
           'mostly_lost', 'num_false_positives', 'num_misses',
           'num_switches', 'num_fragmentations',
           'mota', 'motp', 'num_objects']

summary = mh.compute_many(accs, names=names,
                          metrics=metrics, generate_overall=True)

# Turn raw counts into rates (optional)

div = {'num_objects'       : ['num_false_positives', 'num_misses',
                              'num_switches', 'num_fragmentations'],
       'num_unique_objects': ['mostly_tracked', 'partially_tracked',
                              'mostly_lost']}

for denom, numer_list in div.items():
    for numer in numer_list:
        summary[numer] = summary[numer] / summary[denom]

fmt = mh.formatters.copy()
for k in ('num_false_positives','num_misses','num_switches',
          'num_fragmentations','mostly_tracked','partially_tracked',
          'mostly_lost'):
    fmt[k] = fmt['mota']                  # percentage format

print(mm.io.render_summary(summary, formatters=fmt,
                           namemap=mm.io.motchallenge_metric_names))

metrics = mm.metrics.motchallenge_metrics + ['num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))



