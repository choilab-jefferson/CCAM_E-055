#!/usr/bin/env python3
import glob
import re
import os.path as osp

classes = [
           "WaveHand",
           "DrinkFromBottle",
           "AnswerPhone",
           "Clapping",
           "TightLace",
           "SitDownStandUp",
           "Stationary",
           "ReadWatch",
           "Bow",
           "ComingIn",
           "GoingOut",
           "G230",
           ]
with open("florence_ccam.json", "w") as f_out:
  print("""{
      "categories": [
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
          "g230"
      ],
      "annotations": {""", file=f_out)
  endline = ",\n"
  files = glob.glob("../Florence_3d_actions/*.avi")+glob.glob("../ccam_actions/*.mp4")
  for f in files:
    fname = osp.basename(f)
    if f.find('Florence') >= 0:
      ids = list(map(int, re.findall('\d+', fname)))
      cid = ids[3] - 1
    else:
      names = fname.split("_")
      if fname.find("act")>=0 or fname.find("start")>=0:
        cid = classes.index(names[0])
      elif fname.find("transition1")>=0 and names[0] == "SitDownStandUp":
        cid = classes.index(names[0])+1
      elif fname.find("comingin")>=0:
        cid = 9
      elif fname.find("goingout")>=0:
        cid = 10
      elif fname.find("steady")>=0:
        cid = 6
      else:
        continue
        #cid = -1
    if f == files[-1]:
      endline = "\n"
    print(f'        "{fname}": {{\n          "category_id": {cid}\n        }}', end=endline, file=f_out)
  print("""
    }
}""", file=f_out)
