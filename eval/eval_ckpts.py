import threading
import glob
import os
import queue
import time
import random
from datetime import datetime
import subprocess
import signal
import yaml
import click

config = {
    'gpu_index': 0,
    'model_path': '',
}
base_cmd = '''
export CUDA_VISIBLE_DEVICES='{gpu_index}'

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model {model_path} \
    --chat_template_name default \
    --system_prompt_name disabled \
    --output_dir {output_dir} \
    --tensor_parallel_size {tp} \
    --data_id zwhe99/aime90 \
    --bf16 True \
    --split 2024 \
    --max_model_len 32768 \
    --temperature {temperature} \
    --top_p {top_p} \
    --n 16

'''

def worker_fn(worker_id, task_queue, tp, temperature, top_p, timeout):
    while True:
        try:
            task = task_queue.get(timeout=10) 
        except queue.Empty:
            print(f"Worker-{worker_id}: no job, sleeping")
            time.sleep(10)
            continue
        try:
            print(f"Worker-{worker_id} received {task}")
            task['config']['gpu_index'] = ','.join([str(worker_id + i) for i in range(tp)])
            task['config']['output_dir'] = 'result/' + '_'.join(task['config']['model_path'].split('/')[-2:])
            task['config']['tp'] = tp
            task['config']['temperature'] = temperature
            task['config']['top_p'] = top_p
            cmd = task['base_cmd'].format(**task['config'])
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    preexec_fn=os.setsid,
                )
                # Generous timeout: 2 hour per job; adjust as needed
                ret = proc.wait(timeout=timeout)
                print(f"Worker-{worker_id} finished with exit code {ret}")
            except subprocess.TimeoutExpired:
                print(f"Worker-{worker_id} timeout, killing job")
                try:
                    if proc is not None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception as ke:
                    print(f"Worker-{worker_id} kill error: {ke}")
            except Exception as se:
                print(f"Worker-{worker_id} subprocess error: {se}")
        except Exception as e:
            print(f"Worker-{worker_id} ERROR: {e}")
        finally:
            task_queue.task_done()
            

@click.command()
@click.option('-c', '--config-path', default='config/oss.yaml', help='Path to the configuration YAML file.')
def main(config_path):
    config = yaml.safe_load(open(config_path))

    task_queue = queue.Queue()

    gpu_index = config.get('gpu_index', [0])
    tp = config.get('tp', 1)
    test_base_path = config.get('test_base_path', [])
    temperature = config.get('temperature', 0.6)
    top_p = config.get('top_p', 0.95)
    timeout = config.get('timeout', 7200)  # default 2 hours per job
    
    workers = []

    for i in gpu_index[::tp]:
        t = threading.Thread(target=worker_fn, args=(i,task_queue,tp,temperature,top_p,timeout))
        t.start()
        workers.append(t)


    done_path = set()

    while True:
        test_paths = []
        for test_base_path_tmp in test_base_path:
            paths = glob.glob(test_base_path_tmp + '/check*')
            paths = sorted(paths, key=lambda x:int(x.split('-')[-1]))
            # paths = paths[-3:][::-1]
            paths = paths[::-1]
            for tmp_path in paths:
                if tmp_path in done_path:
                    continue
                done_path.add(tmp_path)
                test_paths.append(tmp_path)
        tasks = []
        for path in test_paths:
            tmp_task = {
                    'config': {
                        'gpu_index': '',
                        'model_path': path,
                        },
                    'base_cmd': base_cmd,
                    }
            print(tmp_task)
            tasks.append(tmp_task)

        for task in tasks:
            task_queue.put(task)

        if tasks:
            task_queue.join()
        else:
            time.sleep(5)


if __name__ == '__main__':
    main()
