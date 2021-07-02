#!/usr/bin/env python
import os
import os.path as osp
import glob
import cv2
import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
inference_recognizer(model, 'demo/demo.mp4', 'demo/label_map_k400.txt')

"""
# Action Recognition

"""
label_map = "demo/label_map_k400.txt"

# config = "configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py"
# # config = "configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth"


# config = "configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py"
# config = "configs/recognition/r2plus1d/r2plus1d_r34_video_inference_8x8x1_180e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb_20200826-ab35a529.pth"


# config = "configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py"
# config = "configs/recognition/slowonly/slowonly_r50_video_inference_4x16x1_256e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth"


config = "configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py"
config = "configs/recognition/slowfast/slowfast_r50_video_inference_4x16x1_256e_kinetics400_rgb.py"
checkpoint = "https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth"


# config = "configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py"
# config = "configs/recognition/tsm/tsm_r50_video_inference_1x1x8_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth"



# config = "configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"

# config = "configs/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth"

# config = "configs/recognition/tsn/tsn_r50_video_imgaug_1x1x8_100e_kinetics400_rgb.py"
# config = "configs/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb_20200703-0f19175f.pth"

# config = "configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py" 
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth"

# label_map = "../../kinetics_600_labels.csv.txt"
# config = "configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015-4db3c461.pth"

# label_map = "../../kinetics_700_labels.csv.txt"
# config = "configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb.py"
# checkpoint = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015-e381a6c7.pth"

device = torch.device('cuda:0')
# build the recognizer from a config file and checkpoint file/url
model = init_recognizer(
    config,
    checkpoint,
    device=device,
    use_frames=False)

# e.g. use ('backbone', ) to return backbone feature
output_layer_names = None

video_list = glob.glob('/ccam_actions/*.mp4')
for i, video_path in enumerate(video_list):
  print(i+1, video_path)
  results = inference_recognizer(model, video_path, label_map, use_frames=False)
  
  print(' The top-5 labels with corresponding scores are:')
  for result in results:
    print(f'    {result[0]:30}: {result[1]:0.2}')
  print()
