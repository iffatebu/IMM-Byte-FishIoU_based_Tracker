# -*- coding: utf-8 -*-
"""
Script: demo_updated_all_singleClassTrack.py

Needed to change: 

????? Parser: "--exp_file"- can change here the directory of YOLOX file
              "--ckpt"- can change the directory of weights file
              "--path"- change the directory for images

????? def run(exp, opt): 
    if opt.save_result: vis_dir = change the directory with tracked visualize
                        result_root = change the name of the folder where .txt file saved
@author: ie93
"""

# encoding=utf-8

import argparse
import os
import time
from collections import defaultdict

import cv2
import torch
from loguru import logger

# from trackers.ocsort_tracker.ocsort import MCOCSort
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import ByteTracker
from yolox.tracking_utils.timer import Timer
#from yolox.utils import fuse_model, get_model_info, post_process
from yolox.utils import fuse_model, get_model_info, postprocess
#from yolox.utils.visualize import plot_tracking_sc, plot_tracking_mc, plot_tracking_ocsort
from yolox.utils.visualize import plot_tracking
from pathlib import Path


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


    

def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("demo",
                        default="video",  # image/ images/videos
                        help="demo type, eg. image, video, videos, and webcam")                         
    
    parser.add_argument("--tracker",
                        type=str,
                        default="byte",
                        help="byte | oc")
    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")
    # parser.add_argument("--reid",
    #                     type=bool,
    #                     default=False,  # True | False
    #                     help="")
    parser.add_argument("-debug",
                        type=bool,
                        default=True,  # True
                        help="")
    
    #--------------Multi-GPU settings for inference
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to use")
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--gpu_rank", type=int, default=0, help="Which chunk of videos to process")

    ## ----- object classes
    parser.add_argument("--n_classes",
                        type=int,
                        default=1,
                        help="")  # number of object classes
    parser.add_argument("--class_names",
                        type=str,
                        default="fish",
                        help="")

    ## ----- exp file, eg: yolox_x_ablation.py
    parser.add_argument("-f",
                        "--exp_file",
                        #default="/scratch/morrill/users/ie93/ByteTrack/YOLOX_outputs/yolox_x_ablation/GFISHER_Whole_testConf_0.01_nms_0.5/YOLOX_X/yolox_x_ablation.py",
                        default="/scratch/morrill/users/ie93/ByteTrack_old/exps/example/mot/Iffat_model_For410Train/yolox_x_ablation.py",
                        type=str,
                        help="pls input your experiment description file")

    ## ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
    parser.add_argument("-c",
                        "--ckpt",
                        default="../pretrained/best_ckpt.pth.tar",
                        type=str,
                        help="ckpt for eval")

    parser.add_argument("--task",
                        type=str,
                        default="track",
                        help="Task mode: track or detect")

    ## ----- videos dir path
    parser.add_argument("--video_dir",
                        type=str,
                        default="../videos",
                        help="")

    ## "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--path",
                        #default="./videos/761901329_cam1_1.mp4",
                        #default="/scratch/morrill/users/ie93/ByteTrack/images/sample_test1",
                        default="/scratch/morrill/users/ie93/ByteTrack_old/datasets/mot/test/Sample_9",
                        help="path to images or video")

    ## ----- Web camera's id
    parser.add_argument("--camid",
                        type=int,
                        default=0,
                        help="webcam demo camera id")
    parser.add_argument("--save_result",
                        #type=bool,
                        #default=True,
                        action="store_true",
                        help="whether to save the inference result of image/video")

    parser.add_argument("--device",
                        default="gpu",
                        type=str,
                        help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf",
                        default=None,
                        type=float,
                        help="test conf")
    parser.add_argument("--nms",
                        default=None,
                        type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize",
                        default=None,
                        type=int,
                        help="test img size")
    parser.add_argument("--fp16",
                        dest="fp16",
                        default=False,  # False
                        action="store_true",
                        help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse",
                        dest="fuse",
                        default=False,
                        action="store_true",
                        help="Fuse conv and bn for testing.")
    parser.add_argument("--trt",
                        dest="trt",
                        default=False,
                        action="store_true",
                        help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_thresh",
                        type=float,
                        default=0.4,    #>>>> recently 0.6 -> 0.4 #------- change, earlier used 0.5, lower value fish may have weaker detections
                        help="detection confidence threshold")
    parser.add_argument("--iou_thresh",
                        type=float,
                        default=0.3,   #------- change, earlier used 0.3
                        help="the iou threshold in Sort for matching")
    parser.add_argument("--match_thresh",
                        type=float,
                        default=0.9,    #>>>>>> change 0.8 -> 0.9, earlier used 0.8, lower value fish move far between frames, IoU is naturally lower
                        help="matching threshold for tracking")
    parser.add_argument("--fps",
                        type=int,
                        default=5,  # orig 30
                        help="the frames count per second")
    parser.add_argument("--track_buffer",
                        type=int,
                        default=10,  # orig track_buffer = 30, then kept it 10 for few days and got improvement
                        help="the frames for keep lost tracks")
    parser.add_argument('--min-box-area',
                        type=float,
                        default=1000,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20",
                         dest="mot20",
                         default=False,
                         action="store_true",
                         help="test mot20.")

    return parser



def get_image_list(path):
    """
    :param path:
    :return:
    """
    image_names = []
    for main_dir, sub_dir, file_name_list in os.walk(path):
        for file_name in file_name_list:
            apath = os.path.join(main_dir, file_name)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results_dict(f_path,
                       results_dict,
                       data_type,
                       num_classes=1):
    """
    :param f_path:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    # if data_type == 'mot':
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
    # elif data_type == 'kitti':
    #     save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    # else:
    #     raise ValueError(data_type)

    with open(f_path, "w", encoding="utf-8") as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for fr_id, tlwhs, track_ids in cls_results:  # fr_id starts from 1
                if data_type == 'kitti':
                    fr_id -= 1

                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=fr_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              cls_id=cls_id)
                    # if fr_id == 1:
                    #     print(line)

                    f.write(line)
                    # f.flush()

    logger.info('Save results to {}.\n'.format(f_path))


def write_results(file_path, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(file_path, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)

    logger.info('save results to {}'.format(file_path))


class Predictor(object):
    def __init__(self,
                 model,
                 exp,
                 trt_file=None,
                 decoder=None,
                 device="cpu",
                 fp16=False,
                 reid=False):
        """
        :param model:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        :param reid:
        """
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.n_classes
        self.conf_thresh = exp.test_conf
        #self.nms_thresh = exp.nms_thresh
        self.nms_thresh = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.reid = reid

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        """
        :param img:
        :param timer:
        :return:
        """
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.mean, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()

            ## ----- forward
            outputs = self.model.forward(img)
            ## -----

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            if self.reid:
                outputs, feature_map = outputs[0], outputs[1]
                outputs = postprocess(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            else:
                if isinstance(outputs, tuple):
                    outputs, feature_map = outputs[0], outputs[1]
                outputs = postprocess(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if self.reid:
            return outputs, feature_map, img_info
        else:
            return outputs, img_info


def image_demo(predictor, vis_folder, result_root, path, current_time, save_result):
    """
    Here output will get as video with full image frames
    :param predictor:
    :param vis_folder:
    :param path:
    :param current_time:
    :param save_result:
    :return:
    """
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]

    files.sort()
    tracker = ByteTracker(opt, frame_rate=opt.fps) #->>>>>>>>>>>>> change
    timer = Timer()
    frame_id = 0
    results = []
    vid_writer = None          # NEW
    vid_save_path = None       # NEW

    for image_name in files:
        if frame_id % 30 == 0:
            if frame_id != 0:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    30.0))

        outputs, img_info = predictor.inference(image_name, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                #vertical = tlwh[2] / tlwh[3] > 1.6
                # if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            # online_im = plot_tracking_sc(img_info['raw_img'],
            #                              online_tlwhs,
            #                              online_ids,
            #                              frame_id=frame_id + 1,
            #                              fps=1.0 / timer.average_time)
            
            online_im = plot_tracking(img_info['raw_img'],
                                         online_tlwhs,
                                         online_ids,
                                         frame_id=frame_id + 1,
                                         fps=1.0 / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            if vid_writer is None:  # set up the writer once, using the first frame's size
                save_folder = os.path.join(vis_folder,
                                           time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                os.makedirs(save_folder, exist_ok=True)

                p = Path(path).resolve()
                seq_name = p.parent.name if p.name.lower() == "img1" else p.stem
                vid_save_path = os.path.join(save_folder, f"{seq_name}.mp4")

                h, w = online_im.shape[:2]
                vid_writer = cv2.VideoWriter(vid_save_path,
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             opt.fps,
                                             (w, h))
            vid_writer.write(online_im)
        # ch = cv2.waitKey(0)

        frame_id += 1
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break
    # # loop ended - save once per video / webcam run 
    # base = os.path.splitext(os.path.basename(opt.path if opt.demo == "video" else "camera"))[0]
    # result_filename = os.path.join(result_root, base + ".txt")
    # If we are inside ".../<seq>/img1", take the parent folder's name.
    
    # """

    if vid_writer is not None:  # NEW
        vid_writer.release()
        logger.info("Video saved to {}.".format(vid_save_path))

    # This was the portion of (from p until write_results) going to change with IMM diagnositc save
    p = Path(path).resolve()
    seq_name = p.parent.name if p.name.lower() == "img1" else p.stem

    result_filename = os.path.join(result_root, f"{seq_name}.txt")
    write_results(result_filename, results)

    # >>>>>>>>>>> new add until def track_video function

    # -- IMM diagnostic save for CA ------------------------------
    if hasattr(tracker, 'logger') and tracker.logger is not None:
        tracker.logger.save(video_name=seq_name)

        summaries = tracker.logger.summarize_tracks()
        cv_dom   = sum(1 for s in summaries if s['cv_dominant'])
        ca_dom   = sum(1 for s in summaries if not s['cv_dominant'])
        needs_ct = sum(1 for s in summaries if s['needs_CT'])

        print(f"\n[IMM SUMMARY: CV+CA] seq={seq_name}")
        print(f"  total tracks : {len(summaries)}")
        print(f"  CV dominant  : {cv_dom}")
        print(f"  CA dominant  : {ca_dom}")
        print(f"  needs CT     : {needs_ct}  <- mu flipping >30% frames")

        tracker.logger.frame_log.clear()
        tracker.logger.track_acc.clear()
    # -----------------------------------------------------


#     # -- IMM diagnostic save for CT ------------------------------
#     if hasattr(tracker, 'logger') and tracker.logger is not None:
#         tracker.logger.save(video_name=seq_name)

#         summaries  = tracker.logger.summarize_tracks()
#         cv_dom     = sum(1 for s in summaries if s['cv_dominant'])
#         ct_dom     = sum(1 for s in summaries if s['ct_dominant'])
#         mixed      = sum(1 for s in summaries if s['mixed'])

#         print(f"\n[IMM SUMMARY: CV+CT] seq={seq_name}")
#         print(f"  total tracks : {len(summaries)}")
#         print(f"  CV dominant  : {cv_dom}   <- fish gliding straight")
#         print(f"  CT dominant  : {ct_dom}   <- fish turning/curving")
#         print(f"  mixed        : {mixed}    <- alternating, watch omega")

#         tracker.logger.frame_log.clear()
#         tracker.logger.track_acc.clear()
# # -----------------------------------------------------

def track_video(predictor, cap, vid_save_path, predict_path ,opt):
    """
    online or offline tracking
    :param predictor:
    :param cap:
    :param vid_save_path:
    :param opt:
    :return:
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    vid_save_path = os.path.abspath(vid_save_path)
    vid_writer = cv2.VideoWriter(vid_save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (int(width), int(height)))

    ## ---------- define the tracker
    if opt.tracker == "byte":
        tracker = ByteTracker(opt, frame_rate=5)
    # elif opt.tracker == "oc":
    #     tracker = MCOCSort(class_names=opt.class_names,
    #                        det_thresh=opt.track_thresh,
    #                        iou_thresh=opt.iou_thresh,
    #                        max_age=opt.track_buffer)
    ## ----------

    ## ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    net_size = exp.test_size

    timer = Timer()

    frame_id = 0
    results = []

    while True:
        if frame_id % 30 == 0:  # logging per 30 frames
            if frame_id != 0:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    n_frames,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    n_frames,
                                    30.0))

        ## ----- read the video
        ret_val, frame = cap.read()

        if ret_val:
            # if opt.reid:
            #     outputs, feature_map, img_info = predictor.inference(frame, timer)
            # else:
            outputs, img_info = predictor.inference(frame, timer)

            dets = outputs[0]

            if dets is not None:
                ## ----- update the frame
                img_size = [img_info['height'], img_info['width']]
                online_targets = tracker.update(dets, img_size, exp.test_size)

                if opt.tracker == "byte":
                    # if opt.reid:
                    #     online_dict = tracker.update_mcmot_emb(dets,
                    #                                            feature_map,
                    #                                            img_size,
                    #                                            exp.test_size)
                    # else:
                    online_dict = tracker.update_mcmot_byte(dets, img_size, net_size)
                        # online_dict = tracker.update_byte_nk(dets, img_size, net_size)
                        # online_dict = tracker.update_byte_enhance2(dets, img_size, net_size)
                        # online_dict = tracker.update_oc_enhance2(dets, img_size, net_size)

                elif opt.tracker == "oc":
                    online_dict = tracker.update_frame(dets, img_size, exp.test_size)

                ## ----- plot single-class multi-object tracking results
                if tracker.n_classes == 1:
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for track in online_targets:
                    # for track in online_dict:
                        tlwh = track.tlwh
                        tid = track.track_id

                        # vertical = tlwh[2] / tlwh[3] > 1.6
                        # vertical = tlwh[2] / tlwh[3] > 1.6
			         
                        # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:

                        if tlwh[2] * tlwh[3] > opt.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(track.score)
                    
                    if frame_id % 6 == 0:
                        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

                    timer.toc()
                    # online_img = plot_tracking_sc(img_info['raw_img'],
                    #                               online_tlwhs,
                    #                               online_ids,
                    #                               frame_id=frame_id + 1,
                    #                               fps=1.0 / timer.average_time)
                    
                    online_img = plot_tracking(img_info['raw_img'],
                                                  online_tlwhs,
                                                  online_ids,
                                                  frame_id=frame_id + 1,
                                                  fps=1.0 / timer.average_time)

                ## ----- plot multi-class multi-object tracking results
                elif tracker.n_classes > 1:
                    if opt.tracker == "byte":
                        ## ---------- aggregate current frame's results for each object class
                        online_tlwhs_dict = defaultdict(list)
                        online_tr_ids_dict = defaultdict(list)
                        for cls_id in range(tracker.n_classes):  # process each object class
                            online_targets = online_dict[cls_id]
                            for track in online_targets:
                                online_tlwhs_dict[cls_id].append(track.tlwh)
                                online_tr_ids_dict[cls_id].append(track.track_id)

                        timer.toc()
                        # online_img = plot_tracking_mc(img=img_info['raw_img'],
                        #                               tlwhs_dict=online_tlwhs_dict,
                        #                               obj_ids_dict=online_tr_ids_dict,
                        #                               num_classes=tracker.n_classes,
                        #                               frame_id=frame_id + 1,
                        #                               fps=1.0 / timer.average_time,
                        #                               id2cls=id2cls)
                        online_img = plot_tracking(img=img_info['raw_img'],
                                                      tlwhs_dict=online_tlwhs_dict,
                                                      obj_ids_dict=online_tr_ids_dict,
                                                      num_classes=tracker.n_classes,
                                                      frame_id=frame_id + 1,
                                                      fps=1.0 / timer.average_time,
                                                      id2cls=id2cls)
                    elif opt.tracker == "oc":
                        timer.toc()
                        # online_img = plot_tracking(img=img_info['raw_img'],
                        #                                   tracks_dict=online_dict,
                        #                                   frame_id=frame_id + 1,
                        #                                   fps=1.0 / timer.average_time,
                        #                                   id2cls=id2cls)
                        online_img = plot_tracking(img=img_info['raw_img'],
                                                          tracks_dict=online_dict,
                                                          frame_id=frame_id + 1,
                                                          fps=1.0 / timer.average_time,
                                                          id2cls=id2cls)
            else:
                timer.toc()
                online_img = img_info['raw_img']

            if opt.save_result:
                vid_writer.write(online_img)

            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            print("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1
    
    write_results(predict_path, results)
    print("{:s} saved.".format(vid_save_path))


def imageflow_demo(predictor, vis_dir, result_root, current_time, opt):
    """
    :param predictor:
    :param vis_dir:
    :param current_time:
    :param opt:
    :return:
    """
    if opt.demo == "videos":
        if os.path.isdir(opt.video_dir):
            mp4_path_list = [opt.video_dir + "/" + x for x in os.listdir(opt.video_dir)
                             if x.endswith(".mp4")]
            mp4_path_list.sort()
            if len(mp4_path_list) == 0:
                logger.error("empty mp4 video list.")
                exit(-1)

            for video_path in mp4_path_list:
                if os.path.isfile(video_path):
                    video_name = os.path.split(video_path)[-1][:-4]
                    print("\nStart tracking video {:s} offline...".format(video_name))

                    ## ----- video capture
                    cap = cv2.VideoCapture(video_path)
                    ## -----

                    save_dir = os.path.join(vis_dir, video_name)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    current_time = time.localtime()
                    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_path = os.path.join(save_dir, current_time + ".mp4")
                    
                    

                    ## ---------- Get tracking results
                    track_video(predictor, cap, save_path, opt)
                    ## ----------

                    print("{:s} tracking offline done.".format(video_name))

    elif opt.demo == "video":
        opt.path = os.path.abspath(opt.path)

        if os.path.isfile(opt.path):
            video_name = opt.path.split("/")[-1][:-4]
            print("Start tracking video {:s} offline...".format(video_name))

            if not os.path.isfile(opt.path):
                logger.error("invalid path: {:s}, exit now!".format(opt.path))
                exit(-1)

            ## ----- video capture
            cap = cv2.VideoCapture(opt.path)
            ## -----

            save_dir = os.path.join(vis_dir, video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            current_time = time.localtime()
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_path = os.path.join(save_dir, current_time + ".mp4")
            
            # loop ended - save once per video / webcam run 
            base = os.path.splitext(os.path.basename(opt.path if opt.demo == "video" else "camera"))[0]
            result_filename = os.path.join(result_root, base + ".txt")
            
            # if not os.path.isdir(result_filename):
            #     os.makedirs(result_filename)
            
            # result_path = os.path.join(result_filename, current_time + ".mp4")

            ## ---------- Get tracking results
            track_video(predictor, cap, save_path, result_filename, opt)
            ## ----------

            print("{:s} tracking done offline.".format(video_name))
        else:
            logger.error("invalid video path: {:s}, exit now!".format(opt.path))
            exit(-1)

    elif opt.demo == "camera":
        if os.path.isfile(opt.path):
            cap = cv2.VideoCapture(opt.camid)
            video_name = opt.path.split("/")[-1][:-4]
            save_dir = os.path.join(vis_dir, video_name)
            save_path = os.path.join(save_dir, "camera.mp4")


def list_img_sequences(root_dir, gpu_rank=0, total_gpus=1):
    """
    Return a list of Paths, each pointing to an 'img1' folder.
    A folder qualifies if it contains at least one .jpg/.png.

    gpu_rank and total_gpus are used to split the list across GPUs.

    Example result:
        [Path('.../SC4-camera3_04-05-21_23-05-44.000/img1'),
         Path('.../761901359_cam4_2/img1'), ...]
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    seqs = []
    for seq_dir in sorted(root.iterdir()):
        img1 = seq_dir / "img1"
        print("img1 path: ", img1)
        if img1.is_dir():
            has_img = any(img1.glob("*.jpg")) or any(img1.glob("*.png"))
            if has_img:
                seqs.append(img1)
    
    # Split across GPUs
    chunk = [v for i, v in enumerate(seqs) if i % total_gpus == gpu_rank]
    
    print(f"[GPU {gpu_rank}] Assigned {len(chunk)}/{len(seqs)} sequences: ")
    for s in chunk:
        print(f"  {s}")
    
    return chunk


def run(exp, opt):
    """
    :param exp:
    :param opt:
    :return:
    """
    if not opt.experiment_name:
        opt.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, opt.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if opt.save_result:
        vis_dir = os.path.join(file_name, "Original_singleClss_410TrainVideo/vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        # NEW: root folder for all demo txt outputs
        result_root = os.path.join(file_name, "Original_singleClss_410TrainVideo/results")
        os.makedirs(result_root, exist_ok=True)

    if opt.trt:
        opt.device = "gpu"

    logger.info("Args: {}".format(opt))
    if opt.conf is not None:
        exp.test_conf = opt.conf
    if opt.nms is not None:
        exp.nms_thresh = opt.nms
    if opt.tsize is not None:
        exp.test_size = (opt.tsize, opt.tsize)

    ## ---------- whether to do ReID
    # if hasattr(exp, "reid"):
    #     exp.reid = opt.reid

    ## ----- Define the network
    net = exp.get_model()
    if not opt.debug:
        logger.info("Model Summary: {}".format(get_model_info(net, exp.test_size)))
    if opt.device == "gpu":
        net.cuda()
    net.eval()
    ## -----

    if not opt.trt:
        if opt.ckpt is None:
            ckpt_file_path = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file_path = opt.ckpt
        ckpt_file_path = os.path.abspath(ckpt_file_path)

        logger.info("Loading checkpoint...")
        ckpt = torch.load(ckpt_file_path, map_location="cpu")

        # load the model state dict
        net.load_state_dict(ckpt["model"])
        logger.info("Checkpoint {:s} loaded done.".format(ckpt_file_path))

    if opt.fuse:
        logger.info("\tFusing model...")
        net = fuse_model(net)

    if opt.fp16:
        net = net.half()  # to FP16

    if opt.trt:
        assert not opt.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), \
            "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        net.head.decode_in_inference = False
        decoder = net.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    ## ---------- Define the predictor
    # predictor = Predictor(net, exp, trt_file, decoder, opt.device, opt.fp16, opt.reid)
    predictor = Predictor(net, exp, trt_file, decoder, opt.device, opt.fp16)
    ## ----------

    current_time = time.localtime()
    if opt.demo == "image":
        image_demo(predictor, vis_dir, result_root, opt.path, current_time, opt.save_result)
    
    elif opt.demo == "images":            # <<< NEW branch
        seq_list = list_img_sequences(opt.path, gpu_rank=opt.gpu_rank, total_gpus=opt.total_gpus)
        if not seq_list:
            logger.error(f"No 'img1' folders found under {opt.path}")
            return
    
        for img1_path in seq_list:
            
            
            seq_name = img1_path.parent.name        # folder name w/o 'img1'
            logger.info(f"=== Processing sequence {seq_name} ===")
    
            ## per-sequence visualisation directory
            seq_vis_dir = os.path.join(vis_dir, seq_name)
            os.makedirs(seq_vis_dir, exist_ok=True)
    
            ## call the existing image_demo on that *one* img1 folder
            image_demo(predictor,
                       vis_folder=seq_vis_dir,
                       result_root=result_root,
                       path=str(img1_path),          # '<seq>/img1'
                       current_time=time.localtime(),
                       save_result=opt.save_result)
            

    elif opt.demo == "video" or opt.demo == "videos" or opt.demo == "webcam":
        imageflow_demo(predictor, vis_dir, result_root, current_time, opt)


if __name__ == "__main__":
    opt = make_parser().parse_args()
    exp = get_exp(opt.exp_file, opt.name)

    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    exp.class_names = class_names
    exp.n_classes = len(exp.class_names)
    print("Number of classes: ", exp.n_classes)

    ## ----- run the tracking
    run(exp, opt)