#!/usr/bin/env python3
# Human Action Classification Pipeline on ROS
#
# Wookjin Choi <wchoi@vsu.edu>
# 08/02/2021
import os
import json
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from mmskeleton.utils import call_obj, load_checkpoint, cache_checkpoint
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager
from collections import deque
from operator import itemgetter

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
    try:
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

            if n_frame > 0 and time_synchronized() - t1 > 0:
                print(f"-- Pose FPS: {n_frame/(time_synchronized() - t1):0.2f}")
                n_frame = 0
                t1 = time_synchronized()
    except KeyboardInterrupt:
        print("Shutting down pose worker")
    finally:
        print("pose worker is off")


def action_worker(poses, results, gpu, cfg):
    worker_id = current_process()._identity[0] - 1
    global action_recognizers
    if worker_id not in action_recognizers:
        action_recognizers[worker_id] = init_action_recognizer(cfg, device=gpu)
    print("action worker is on")
    try:
        t1 = time_synchronized()
        t2 = time_synchronized()
        pose_queue = deque(maxlen=17)
        score_cache = deque()
        scores_sum = 0
        n_frame = 0
        while True:
            pose = poses.get()
            if pose is None:
                return
            pose_queue.append(pose)

            cur_windows = []
            if len(pose_queue) ==  17:
                cur_windows = list(pose_queue)

                scores = inference_action_recognizer(action_recognizers[worker_id], cur_windows)
                
                score_cache.append(scores[0])
                scores_sum += scores[0]

                if len(score_cache) == cfg.average_size:
                    scores_avg = scores_sum / cfg.average_size
                    num_selected_labels = min(len(cfg.label), 5)

                    scores_tuples = tuple(zip(cfg.label, scores_avg))
                    scores_sorted = sorted(
                        scores_tuples, key=itemgetter(1), reverse=True)
                    result = scores_sorted[:num_selected_labels]

                    results.put(result)
                    n_frame += 1
                    scores_sum -= score_cache.popleft()
            
            if n_frame > 0 and time_synchronized() - t1 > 0:
                print(f"-- Action FPS: {n_frame/(time_synchronized() - t1):0.2f}")
                n_frame = 0
                t1 = time_synchronized()
            if cfg.inference_fps > 0:
                # add a limiter for actual inference fps <= inference_fps
                sleep_time = 1 / cfg.inference_fps - (time_synchronized() - t2)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                t2 = time_synchronized()
    except KeyboardInterrupt:
        print("Shutting down action worker")
    finally:
        print("action worker is off")



class ActionRecognitionPipeline():
    def __init__(self,
                 cfg,
                 video_max_length=100):

        # initialization
        cache_checkpoint(cfg.detection_cfg.checkpoint_file)
        cache_checkpoint(cfg.estimation_cfg.checkpoint_file)

        cfg.label = categories

        self.is_run = True
        self.frame_index = 0
        self.gpus = cfg.gpus
        self.detection_cfg = cfg.detection_cfg
        self.estimation_cfg = cfg.estimation_cfg
        self.recognition_cfg = cfg.recognition_cfg
        self.video_categories = categories
        self.inputs = Manager().Queue(video_max_length)
        self.poses = Manager().Queue(video_max_length/4)
        self.results = Manager().Queue(video_max_length)
        self.num_worker = cfg.gpus * cfg.worker_per_gpu

        self.procs = []
        for i in range(self.num_worker):
            p_pose = Process(target=pose_worker, args=(self.inputs, self.poses, i % self.gpus, cfg))
            p_action = Process(target=action_worker, args=(self.poses, self.results, i % self.gpus, cfg))
            self.procs.append((p_pose, p_action))
            p_pose.start()
            p_action.start()
            

    def get_result(self):
        t = None
        try:
            t = self.results.get(False)
        except:
            pass
        if t is not None:
            self.frame_index = 0
        return t

    def stop(self):
        # send end signals
        [self.inputs.popleft() for _ in range(self.inputs.qsize())]
        [self.poses.popleft() for _ in range(self.poses.qsize())]
        for _ in self.procs:
            self.inputs.put((-1, None))
            self.poses.put(None)
        # wait to finish
        for p_pose, p_action in self.procs:
            p_pose.join()
            p_action.join()

    def put_frame(self, frame):
        try:
            self.inputs.put((self.frame_index, frame), False)
            self.frame_index += 1
            if self.frame_index > 16:
                self.frame_index = 0
        except:
            pass