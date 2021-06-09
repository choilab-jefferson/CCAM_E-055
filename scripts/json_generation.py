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
    "annotations": {""")
files = glob.glob("Florence_3d_actions/*.avi")+glob.glob("ccam_actions/*.mp4")
for f in files:
  fname = osp.basename(f)
  if f[0] == 'F':
    ids = list(map(int, re.findall('\d+', fname)))
    cid = ids[3] - 1
  else:
    names = fname.split("_")
    if names[-1] == "comingin.mp4":
      cid = 9
    elif names[-1] == "goingout.mp4":
      cid = 10
    else:
      cid = classes.index(names[0])
  print(f'        "{fname}": {{\n          "category_id": {cid}\n        }},')
print("""\b\b
    }
}""")
