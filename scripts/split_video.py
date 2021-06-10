import glob
import subprocess
import os
import os.path as osp
import numpy as np
import pandas as pd

df = pd.read_csv('../ccam_actions_org/actions.csv') 
print(df)
os.makedirs("../ccam_actions/", exist_ok = True)
for i, action in enumerate(df['Actions']):
    files = glob.glob(f"../ccam_actions_org/{action}*.mp4")
    print(df.loc[i])
    p = 0
    for idx, t in df.loc[i][1:].items():
        if t != t:
            if idx.find("outside")>=0:
                t = 1000
            else:
                continue
        idx = idx.replace("end", "act")
        idx = idx.replace(" ", "_").replace(".","")
        print(idx, p, t)
        for file_name in files:
            print(file_name)
            cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {file_name}"
            fps_str = subprocess.check_output(cmd, shell=True).decode("utf-8") 
            print(fps_str)
            fps = float(fps_str.split("/")[0])

            out_file_name = osp.splitext(osp.basename(file_name))[0]
            p1 = p
            steps = np.linspace(p, t, int((t-p)/(fps+5)))
            if len(steps) > 1:
                steps = steps[1:]
            else:
                steps = [t]
            print(steps, len(steps))
            for t1 in steps:
                cmd = f"ffmpeg -i {file_name} -ss 00:00:{p1/fps:02} -t 00:00:{(t1-p1)/fps:02} ../ccam_actions/{out_file_name}_{idx}_{round(p1)}-{round(t1)}.mp4"
                print(cmd)
                subprocess.call(cmd, shell=True)
                p1 = t1
        p = t

