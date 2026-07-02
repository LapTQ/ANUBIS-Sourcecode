ls_path_cfg = [
    "v219--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2",
    "v220--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--b",
    "v221--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--jm",
    "v222--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--bm",
]
num_models = 10
ls_devices = [2, 2, 2, 2]

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
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]
