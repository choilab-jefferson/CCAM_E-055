# multiview simulation generation from RGBD sequence datasets
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/20/2021

import json
import argparse
import time
import datetime
import sys
import shutil
from os import makedirs
from os.path import isfile, join, exists, split, normpath
import open3d as o3d
from initialize_config import initialize_config
from utility.file import check_folder_structure, get_rgbd_file_lists


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        #user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        # if user_input.lower() == 'y':
        shutil.rmtree(path_folder)
        makedirs(path_folder)
        # else:
        #    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiview dataset generation")
    parser.add_argument("config", help="path to the config file")

    args = parser.parse_args()

    # check folder structure
    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config["path_dataset"])
    assert config is not None

    print("generating multiview RGBD sequences.")

    config_prefix = args.config.replace(".json", "")
    path_dataset = split(normpath(config["path_dataset"]))
    path_output = join(path_dataset[0],path_dataset[1]+"_multiview/")
    path_depth = join(path_output, "depth")
    path_color = join(path_output, "color")
    make_clean_folder(path_output)
    make_clean_folder(path_depth)
    make_clean_folder(path_color)

    # load files and split into subsets
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    [color_files_scene, depth_files_scene] = color_files[0:-1:2], depth_files[0:-1:2]
    [color_files_multiview, depth_files_multiview] = (
        color_files[1:-1:2], depth_files[1:-1:2])

    # copy scene subset
    for color_file, depth_file in zip(color_files_scene, depth_files_scene):
        color_file_output, depth_file_output = (
            color_file.replace(config["path_dataset"], path_output).replace(
                "image", "color").replace("rgb", "color"),
            depth_file.replace(config["path_dataset"], path_output)
        )
        shutil.copy(color_file, color_file_output)
        shutil.copy(depth_file, depth_file_output)

    # copy config
    with open(args.config) as json_file:
        config_scene = json.load(json_file)
        config_scene['path_dataset'] = path_output
        with open(f'{config_prefix}_multiview_scene.json', 'w') as json_file_output:
            json.dump(config_scene, json_file_output)
    sys.stdout.flush()

    # copy multiview fragments
    n_frames = len(color_files_multiview)
    n_frames_div8 = int(n_frames / 8)
    n_frames_frag_div10 = int(config["n_frames_per_fragment"]/10)
    for view in range(0, 4):
        s = view * 2 + 1
        color_files_multiview_fragment, depth_files_multiview_fragment = (
            color_files_multiview[s*n_frames_div8:s *
                                  n_frames_div8+n_frames_frag_div10],
            depth_files_multiview[s*n_frames_div8:s *
                                  n_frames_div8+n_frames_frag_div10]
        )
        # / required to replace path
        path_output_multiview = join(path_output, f"multiview{view}/")
        path_depth_multiview = join(path_output_multiview, "depth")
        path_color_multiview = join(path_output_multiview, "color")
        make_clean_folder(path_output_multiview)
        make_clean_folder(path_depth_multiview)
        make_clean_folder(path_color_multiview)
        for color_file, depth_file in zip(color_files_multiview_fragment, depth_files_multiview_fragment):
            color_file_output, depth_file_output = (
                color_file.replace(config["path_dataset"], path_output_multiview).replace(
                    "image", "color").replace("rgb", "color"),
                depth_file.replace(
                    config["path_dataset"], path_output_multiview)
            )
            shutil.copy(color_file, color_file_output)
            shutil.copy(depth_file, depth_file_output)

        # copy config
        with open(args.config) as json_file:
            config_multiview = json.load(json_file)
            config_multiview['path_dataset'] = path_output_multiview
            with open(f'{config_prefix}_multiview{view}.json', 'w') as json_file_output:
                json.dump(config_multiview, json_file_output)
        sys.stdout.flush()
