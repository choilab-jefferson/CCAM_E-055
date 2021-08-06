#!/usr/bin/env python3
# Human Action Classification Pipeline on ROS for mmaction2
#
# Wookjin Choi <wchoi@vsu.edu>
# 08/06/2021
import time
import torch
import torch.multiprocessing as mp
from mmcv.parallel import collate, scatter
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from multiprocessing import current_process, Process, Manager
from collections import deque
from operator import itemgetter

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

action_recognizers = dict()
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def inference_action_recognizer(action_recognizer, cur_data, device):
    recognition_model, recognition_cfg = action_recognizer

    if next(recognition_model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [device])[0]

    with torch.no_grad():
        scores = recognition_model(return_loss=False, **cur_data)[0]

    return scores


def action_worker(inputs, results, gpus, cfg):
    device = torch.device(cfg.device)
    worker_id = current_process()._identity[0] - 1
    global action_recognizers
    if worker_id not in action_recognizers:
        print(cfg.checkpoint)
        model = init_recognizer(cfg.model_cfg, cfg.checkpoint, device=device)
        action_recognizers[worker_id] = (model, cfg.model_cfg)

    data = dict(img_shape=None, modality='RGB', label=-1)

    # prepare test pipeline from non-camera pipeline
    model_cfg = cfg.model_cfg
    sample_length = 0
    pipeline = model_cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    print("action worker is on")

    try:
        t1 = time_synchronized()
        t2 = time_synchronized()
        score_cache = deque()
        frame_queue = deque(maxlen=sample_length)
        scores_sum = 0
        n_frame = 0
        while True:
            idx, image = inputs.get()
            if image is None:
                return
            frame_queue.append(image[0])

            cur_windows = []
            if len(frame_queue) ==  sample_length:
                cur_windows = list(frame_queue)
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

                cur_data = data.copy()
                cur_data['imgs'] = cur_windows
                cur_data = test_pipeline(cur_data)
                cur_data = collate([cur_data], samples_per_gpu=1)

                scores = inference_action_recognizer(action_recognizers[worker_id], cur_data, device)

                score_cache.append(scores)
                scores_sum += scores

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

            if time_synchronized() - t1 > 1 and n_frame > 0:
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
        self.cfg = cfg

        self.is_run = True
        self.frame_index = 0
        self.gpus = cfg.gpus
        self.recognition_cfg = cfg.model_cfg
        self.video_categories = cfg.label
        self.inputs = Manager().Queue(video_max_length)
        self.results = Manager().Queue(video_max_length)
        self.num_worker = cfg.gpus * cfg.worker_per_gpu

        self.procs = []
        for i in range(self.num_worker):
            #p_pose = Process(target=pose_worker, args=(self.inputs, self.poses, i % self.gpus, cfg))
            p_action = Process(target=action_worker, args=(self.inputs, self.results, i % self.gpus, cfg))
            self.procs.append(p_action)
            #self.procs.append((p_pose, p_action))
            #p_pose.start()
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
        for _ in self.procs:
            self.inputs.put((-1, None))
            #self.poses.put(None)
        # wait to finish
        for p_action in self.procs:
            #p_pose.join()
            p_action.join()

    def put_frame(self, frame):
        try:
            self.inputs.put((self.frame_index, frame), False)
            self.frame_index += 1
            if self.frame_index > 16:
                self.frame_index = 0
        except:
            pass
