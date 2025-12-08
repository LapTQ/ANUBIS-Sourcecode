ls_path_cfg = [
    "v227--satudora_veo3_awlset1set2set3_107cia--split-14-class--nodistinct--2s-15frames--v2--j",
    "v228--satudora_veo3_awlset1set2set3_107cia--split-14-class--nodistinct--2s-15frames--v2--b",
    "v229--satudora_veo3_awlset1set2set3_107cia--split-14-class--nodistinct--2s-15frames--v2--jm",
    "v230--satudora_veo3_awlset1set2set3_107cia--split-14-class--nodistinct--2s-15frames--v2--bm",
]
num_models = 10
ls_devices = [1, 1, 1, 1]

import subprocess
import multiprocessing as mp
import concurrent.futures

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for id_model in range(num_models):
        futures = []
        for name_cfg, device in zip(ls_path_cfg, ls_devices):
            futures.append(
                executor.submit(
                    subprocess.run,
                    args=f"python main.py --config config/fs26/{name_cfg}.yaml --device {device} --overwrite --id_model {id_model}",
                    # cwd="/home/laptq/laptq-fs26-shoplifting-detection/submodules/ProtoGCN",
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]
