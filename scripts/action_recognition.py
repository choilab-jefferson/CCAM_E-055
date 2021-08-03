#!/usr/bin/env python3
# Human Action Classification Pipeline on ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 08/02/2021
import os
import sys
import json
import time
import mmcv
import numpy as np
import ntpath
import torch
import torch.multiprocessing as mp
import argparse
import logging
from collections import OrderedDict
import mmskeleton
from mmskeleton.utils import call_obj, set_attr, get_attr, import_obj, load_checkpoint, cache_checkpoint
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from mmskeleton.datasets import skeleton
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
from mmcv.utils import ProgressBar
from multiprocessing import current_process, Process, Manager
from mmskeleton.processor.recognition import topk_accuracy

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

pose_estimators = dict()
action_recognizers = dict()
categories = [
    "wave",
    "drink from a bottle",
    "answer phone",
    "clap",
    "tight lace",
    "sit down",
    "stand up",
    "read watch",
    "bow",
    "coming in",
    "going out",
    "g230"]


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def init_action_recognizer(cfg, device=None):
    recognition_cfg = cfg.recognition_cfg
    # put model on gpus
    if isinstance(recognition_cfg, list):
        recognition_model = [call_obj(**c) for c in recognition_cfg]
        recognition_model = torch.nn.Sequential(*recognition_model)
    else:
        recognition_model = call_obj(**recognition_cfg)
    checkpoint = cfg.checkpoint
    load_checkpoint(recognition_model, checkpoint, map_location='cpu')
    #recognition_model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        recognition_model = recognition_model.cuda()
    recognition_model.eval()
    action_recognizer = (recognition_model, recognition_cfg)
    return action_recognizer


def inference_action_recognizer(action_recognizer, poses):
    recognition_model, recognition_cfg = action_recognizer

    output = None

    annotations = []
    num_keypoints = 0
    num_frame = 0
    num_track = 2
    for pose in poses:
        if pose["has_return"]:
            num_frame += 1
            num_person = len(pose['joint_preds'])
            assert len(pose['person_bbox']) == num_person

            for j in range(num_person):
                keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                    pose['joint_preds'][j].round().astype(int).tolist(), pose[
                        'joint_scores'][j].tolist())]
                num_keypoints = len(keypoints)
                person_info = dict(
                    person_bbox=pose['person_bbox'][j].round().astype(int)
                    .tolist(),
                    frame_index=pose['frame_index'],
                    id=j,
                    person_id=None,
                    keypoints=keypoints)
                annotations.append(person_info)

    # get data
    data = np.zeros(
        (3, num_keypoints, num_frame, num_track),
        dtype=np.float32)
    i = 0
    for a in annotations:
        person_id = a['id'] if a['person_id'] is None else a['person_id']
        frame_index = a['frame_index']

        if person_id < num_track and frame_index < num_frame:
            i += 1
            data[:, :, frame_index, person_id] = np.array(
                a['keypoints']).transpose()

    if i < 10:
        return None

    data = torch.Tensor([data]).cuda()
    # print(data)
    # print(data.size())
    with torch.no_grad():
        try:
            output = recognition_model(data)
            output = recognition_model(data).data.cpu().numpy()
        except:
            output = None

    return output


def pose_worker(inputs, poses, gpu, cfg):
    worker_id = current_process()._identity[0] - 1
    global pose_estimators
    if worker_id not in pose_estimators:
        pose_estimators[worker_id] = init_pose_estimator(
            cfg.detection_cfg, cfg.estimation_cfg, device=gpu)
    print("pose worker is on")
    t1 = time_synchronized()
    n_frame = 0
    while True:
        idx, image = inputs.get()

        # end signal
        if image is None:
            return

        res_pose = inference_pose_estimator(
            pose_estimators[worker_id], image[0])
        res_pose['frame_index'] = idx

        poses.put(res_pose)
        n_frame += 1
        
        if(time_synchronized() - t1 > 1):
            print(f"-- Pose FPS: {n_frame/(time_synchronized() - t1):0.2f}")
            n_frame = 0
            t1 = time_synchronized()


def action_worker(poses, results, gpu, cfg):
    worker_id = current_process()._identity[0] - 1
    global action_recognizers
    if worker_id not in action_recognizers:
        action_recognizers[worker_id] = init_action_recognizer(cfg, device=gpu)
    print("action worker is on")
    t1 = time_synchronized()
    while True:
        # if image is None: # TODO: add signal to stop
        #     return
        if poses.qsize() > 16:
            print(poses.qsize())
            res_poses = [poses.get() for _ in range(poses.qsize())]
            res_action = inference_action_recognizer(
                action_recognizers[worker_id], res_poses)
            if res_action is not None:
                results.put(res_action)
                print(f"Action Recognition takes {(time_synchronized() - t1):0.2f} secs")
                t1 = time_synchronized()



class ActionRecognitionPipeline():
    def __init__(self,
                 cfg,
                 video_max_length=100):

        # initialization
        cache_checkpoint(cfg.detection_cfg.checkpoint_file)
        cache_checkpoint(cfg.estimation_cfg.checkpoint_file)

        if cfg.category_annotation is None:
            video_categories = dict()
        else:
            with open(cfg.category_annotation) as f:
                video_categories = json.load(f)['annotations']

        self.is_run = True
        self.frame_index = 0
        self.gpus = cfg.gpus
        self.detection_cfg = cfg.detection_cfg
        self.estimation_cfg = cfg.estimation_cfg
        self.recognition_cfg = cfg.recognition_cfg
        self.video_categories = video_categories
        self.inputs = Manager().Queue(video_max_length)
        self.poses = Manager().Queue(video_max_length/4)
        self.results = Manager().Queue(video_max_length)
        self.num_worker = cfg.gpus * cfg.worker_per_gpu

        self.procs = []
        for i in range(self.num_worker):
            p = Process(
                target=pose_worker,
                args=(self.inputs, self.poses, i % self.gpus, cfg))
            self.procs.append(p)
            p.start()
            p = Process(
                target=action_worker,
                args=(self.poses, self.results, i % self.gpus, cfg))
            self.procs.append(p)
            p.start()

    def get_result(self):
        t = None
        try:
            t = self.results.get(False)
        except:
            pass
        if t is not None:
            cat = np.array(categories)
            idx = np.flip(np.argsort(t))[0][0:4]
            print(t)
            print(cat[idx])
            print(t[0,idx])
            self.frame_index = 0

        return 0

    def stop(self):
        # send end signals
        for p in self.procs:
            self.inputs.put((-1, None))
        # wait to finish
        for p in self.procs:
            p.join()

    def put_frame(self, frame):
        self.inputs.put((self.frame_index, frame))
        self.frame_index += 1
        if self.frame_index > 16:
            self.frame_index = 0
