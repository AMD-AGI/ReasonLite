from datetime import datetime
import click
import threading
import itertools
import yaml
import glob
import os
import queue
import time
import random
import time

base_cmd = '''
export CUDA_VISIBLE_DEVICES='{gpu_index}'

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 eval.py
'''

def worker_fn(worker_id, task_queue):
    while 1:
        try:
            task = task_queue.get(timeout=10)
        except queue.Empty:
            print(f"Worker-{worker_id}: no job, sleeping")
            time.sleep(10)
            continue
        try:
            print(f"Worker-{worker_id} received {task}")
            cmd = base_cmd.format(gpu_index=worker_id)
            cmd = cmd.strip() + ' \\\n ' + ' \\\n '.join([f'--{k} {v}' for k, v in task.items()])
            print(cmd)
            os.system(cmd)
            print(f"Worker-{worker_id} finished {task}")
        except Exception as e:
            print(f"Worker-{worker_id} ERROR: {e}")
        finally:
            task_queue.task_done()

    


@click.command()
@click.option('-c', '--config-path', default='config/eval.yaml', help='Path to the configuration YAML file.')
def main(config_path):
    task_queue = queue.Queue()
    
    gpu_index = [0,1,2,3,4,5,6,7]
    workers = []
    
    for i in gpu_index:
        t = threading.Thread(target=worker_fn, args=(i,task_queue))
        t.start()
        workers.append(t)
    
    
    with open(config_path, "r", encoding="utf-8") as f:
        eval_kwargs_all = yaml.safe_load(f)

    tasks = []
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    for eval_kwargs in eval_kwargs_all:
        keys = list(eval_kwargs.keys())
        values_lists = [eval_kwargs[key] if type(eval_kwargs[key]) is list else [eval_kwargs[key]] for key in keys]

        for combination in itertools.product(*values_lists):
            kwargs = dict(zip(keys, combination))
            kwargs['output_dir'] = f'result/{file_name}/{kwargs["base_model"].split("/")[-2]}_{kwargs["base_model"].split("/")[-1]}_' + str(time.time())
            tasks.append(kwargs)
            print(kwargs)

    for task in tasks:
        task_queue.put(task)

    task_queue.join()
    
if __name__ == '__main__':
    main()
