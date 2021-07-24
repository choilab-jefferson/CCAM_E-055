# batch run of multiview simulation 
#
# Wookjin Choi <wchoi@vsu.edu>
# 07/24/2021
import os
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiview simulation batch")
    parser.add_argument("config_dir", help="path to the config files")

    args = parser.parse_args()

    config_files = glob.glob(args.config_dir + "/**/*.json", recursive = True)
    for config in config_files:
        # skip multiview datasets
        if config.find("multiview") >= 0:
            continue
        print(config)
        
        # generate multiview simulation dataset
        os.system(f"python3 multiview_simulation_dataset.py {config}")

        # reconstruct scene and register multiview to the scene
        os.system(f"python3 run_system.py --all --device=cuda:0 {config.replace('.json','_multiview.json')}")
    
