ls_path_cfg = [
    "v259.10_classes.2s-15frames.j.filter_dbscan_95",
    "v260.10_classes.2s-15frames.b.filter_dbscan_95",
    # "v261.10_classes.2s-15frames.jm.filter_dbscan_95",
    # "v262.10_classes.2s-15frames.bm.filter_dbscan_95",
]
num_models = 1
ls_devices = [1, 1]

import concurrent.futures
import subprocess
import sys

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for id_model in range(num_models):
        futures = []
        for name_cfg, device in zip(ls_path_cfg, ls_devices):
            futures.append(
                executor.submit(
                    subprocess.run,
                    args=f"{sys.executable} main.py --config config/fs26/{name_cfg}.yaml --device {device} --overwrite --id_model {id_model}",
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]
