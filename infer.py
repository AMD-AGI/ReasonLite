import requests
import click
import os
from datetime import datetime
import glob
import numpy as np
import pickle
from copy import deepcopy
import tqdm
import yaml
import json
from datasets import load_dataset
from openai import OpenAI
import time
import threading
import queue
import jsonlines
from utils.tools import until_servers_up
from utils.mode_handler import ReasoningModeHandler, Mode
from utils.logging_utils import logger

class vllm_infer():
    def __init__(self, config_path, save_path, chat=True, postprocess=None):
        if chat:
            self.base_url = "http://{localhost}:{port}/v1/chat/completions"
        else:
            self.base_url = "http://{localhost}:{port}/v1/completions"
        self.config = yaml.safe_load(open(config_path))
        self.model_name = self.get_model_name()
        self.localhost = self.config.get('ip', ['localhost'])
        self.world_size = len(self.localhost)
        self.batch_size = self.config.get('batch_size', 768)
        self.mp_num = self.batch_size * len(self.config['gpu_index']) // int(self.config['tp']) * self.world_size
        self.save_path = save_path
        self.postprocess=postprocess

    def get_answer(self, datas):
        gpu_index_part = [self.config['gpu_index'][i:i+self.config['tp']] for i in range(0, len(self.config['gpu_index']), self.config['tp'])]
        ports = [8040 + i[0] for i in gpu_index_part]
        urls = [self.base_url.format(localhost=localhost, port=port) for port in ports for localhost in self.localhost for _ in range(self.batch_size)] 
        self.urls = urls
        datas = [{'model': self.model_name, 'extra': {}, **i} for c, i in enumerate(datas)]
        output = self.mp_infer(datas)
        return output
        

    def mp_infer(self, datas):
        input_queue = queue.Queue()
        output_queue = queue.Queue()

        num_threads = self.mp_num 
        for data in datas:
            input_queue.put(data)
        for _ in range(num_threads):
            input_queue.put(None)
        threads = []
        tmp = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker, args=(input_queue, output_queue, self.urls[_]))
            t.start()
            threads.append(t)
            tmp.append(self.urls[_])
        out = []
        writer = jsonlines.open(self.save_path, mode='a')
        to_write = []
        for i in tqdm.tqdm(range(len(datas))):
            tmp_out = output_queue.get()
            try:
                tmp_out['answer'] = tmp_out['output']['choices'][0]['text']
            except:
                tmp_out['answer'] = None
            if self.postprocess is not None:
                tmp_out = self.postprocess(tmp_out)
            if self.save_path is not None:
                to_write.append(tmp_out)
            else:
                out.append(tmp_out)
            if len(to_write) > 128:
                writer.write_all(to_write)
                to_write = []
        if len(to_write) > 0:
            writer.write_all(to_write)

        for t in threads:
            t.join()

        return out

    def vllm_request(self, data):
        extra = data['extra']
        try:
            r = requests.post(extra['url'], json=data).json()
            data['output'] = r
        except:
            data['output'] = None
            time.sleep(120)
        return data
    
    def worker(self, input_queue, output_queue, urls):
        while True:
            input_data = input_queue.get()
            if input_data is None:
                break
            input_data['extra']['url'] = urls
            result = self.vllm_request(input_data)
            output_queue.put(result)
            input_queue.task_done()
    

    def get_model_name(self):
        for port in range(8040, 8048):
            try:
                openai_api_key = "EMPTY"
                openai_api_base = f"http://localhost:{port}/v1"
                
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )
                
                models = client.models.list()
                model = models.data[0].id
                return model
            except:
                pass
    



@click.command()
@click.option('-c', '--config-path', default='config/oss.yaml', help='Path to the configuration YAML file.')
@click.option('-m', '--mode', default=None, type=click.Choice([m.value for m in Mode], case_sensitive=False))
@click.option('-l', '--local_rank', default=0)
@click.option('-g', '--global_size', default=1)
@click.option('-s', '--save_file', default=None)
def main(config_path, local_rank, global_size, save_file, mode):

    ## update config
    config = yaml.safe_load(open(config_path))
    
    # wait until all servers are up
    if config.get('wait_servers_up', False):
        until_servers_up(config)
    
    base_data_path = config['base_data_path']
    if mode is None:
        mode = str(config.get('mode', Mode.INFER.value))
    else:
        config['mode'] = mode
    mode_enum = Mode.from_str(mode.lower())
    handler = ReasoningModeHandler(config, mode_enum)
    num_answer = config.get('num_answer', 8)
    logger.info(config)

    ## save data path
    if save_file is None:
        save_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_data_name = config.get('new_data_name', handler.data_name)
    save_path = os.path.join(base_data_path, new_data_name, save_file, f'{local_rank}_{global_size}.jsonl')
    os.makedirs(os.path.join(base_data_path, new_data_name, save_file), exist_ok=True)
    logger.info(f'[save data to]: {save_path}')
    

    ## load prompt data
    base_ds = handler.get_prompt_data(num_answer)


    ## get latest data
    latest_ds = handler.get_latest_data(save_file)

    ## prepare vllm input & save finished date
    # vllm input
    vllm_input_datas = []
    # finished data & info
    finished_data_info = {i["index"]: i for i in latest_ds if 'index' in i}
    finished = [] 

    count = 0
    for c, data in enumerate(tqdm.tqdm(base_ds)):
        key = data["index"]
        if key in finished_data_info and finished_data_info[key].get('answer', None) is not None:
            finished.append(finished_data_info[key])
            continue
        vllm_data = handler.preprocess(data)
        if count % global_size == local_rank:
            vllm_input_datas.append(vllm_data)
        count += 1

    logger.info(f'[total num]: {len(base_ds)}\n[latest num]: {len(latest_ds)}\n[finish num]: {len(finished)}\n')
    if local_rank == 0:
        logger.info(f'save data: {save_path}')
        with jsonlines.open(save_path, mode='w') as writer:
            writer.write_all(finished)


    ## vllm infer & postprecess
    postprocess = handler.postprocess

    f = vllm_infer(config_path, chat=False, save_path=save_path, postprocess=postprocess)
    out = f.get_answer(vllm_input_datas)


    logger.info(f'save data: {save_path}')

if __name__ == '__main__':
    main()
