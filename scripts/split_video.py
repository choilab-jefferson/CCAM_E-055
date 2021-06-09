import glob
import subprocess
import os.path as osp
import numpy as np

files = glob.glob("ccam_actions_org/*.mp4")
for file_name in files:
    print(file_name)
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 {file_name}"
    duration = float(subprocess.check_output(cmd, shell=True))
    print(duration)
    seg = duration/4
    if seg > 5:
        seg = duration/5
    start_times = np.arange(0,duration,seg)
    print(start_times)
    time_table = {"comingin":0, "goingout":start_times[-1]}
    for i, t in enumerate(start_times[1:-1]):
        time_table[f"act{i+1}"]=t-0.5
    print(time_table)
    
    out_file_name = osp.splitext(osp.basename(file_name))[0]
    for k, t in time_table.items():
        print(k,t)
        cmd = f"ffmpeg -i {file_name} -ss 00:00:{t:02} -t 00:00:{seg+0.5:02} ccam_actions/{out_file_name}_{k}.mp4"
        subprocess.call(cmd, shell=True)

