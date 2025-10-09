import requests
import click
import os
from loguru import logger
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
import re
from utils.tools import get_last_boxed, get_vote_result
from utils.prompt_func import infer_prompt, vote_prompt, judge_prompt


class vllm_infer():
    def __init__(self, config_path='/llm_training/zihao/2024/inference/vllm_infer/config/base.yaml', chat=True, save_path=None, postprocess=None):
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
        if self.save_path is not None:
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
    

def get_prompt_data(config, mode, num_answer):
    base_data_path = config['base_data_path']
    if mode == 'infer':
        prompt_json_paths = [os.path.join(base_data_path, config.get('data_info_name', 'info.jsonl'))]
    else:
        prompt_path = os.path.join(base_data_path, data_name_dict['infer'])
        prompt_path = sorted(glob.glob(os.path.join(prompt_path, '*')))[-1]
        prompt_json_paths = glob.glob(os.path.join(prompt_path, '*.jsonl'))
    logger.info(f'[load data from]: {prompt_json_paths}')

    base_ds_ori = []
    for prompt_json_paths_tmp in prompt_json_paths:
        base_ds_ori.extend([json.loads(i) for i in open(prompt_json_paths_tmp).readlines()])

    if mode == 'infer':
        base_ds = []
        for c, i in enumerate(base_ds_ori):
            i['index'] = str(i.get('index', c))
            if len(i['index'].split('_')) == 2:
                base_ds.append(i)
            else:
                for j in range(num_answer):
                    tmp = deepcopy(i)
                    tmp['index'] = f"{tmp['index']}_{j}"
                    base_ds.append(tmp)
    elif mode == 'vote':
        data_by_id = {}
        for i in tqdm.tqdm(base_ds_ori):
            index = i['index']
            prompt_index = index.split('_')[0]
            i['info']['index'] = str(prompt_index)
            gt = i['info']['expected_answer']
            answer = get_last_boxed(i['model_output'])
            if prompt_index not in data_by_id:
                data_by_id[prompt_index] = []
            tmp_data = {
                'gt': gt,
                'answer': answer,
                'info': i['info'],
                }
            data_by_id[prompt_index].append(tmp_data)
        check_answer = []
        for c, index in enumerate(data_by_id):
            tmp_data = data_by_id[index]
            answer = [i['answer'] for i in tmp_data]
            check_answer.append({
                'index': index,
                'answer': answer,
                'gt': tmp_data[0]['gt'],
                'info': tmp_data[0]['info'],
                })
        base_ds = check_answer
    elif mode == 'judge_vote':
        vote_info_path = sorted(glob.glob(os.path.join(base_data_path, data_name_dict['vote'], '*', '*.jsonl')))[-1]
        vote_info = [json.loads(i) for i in open(vote_info_path).readlines()]
        vote_info_dict = {i['index']: i for i in vote_info}
        base_ds = []
        for data in base_ds_ori:
            data['info'] = vote_info_dict[data['index'].split('_')[0]]
            base_ds.append(data)
    else:
        base_ds = base_ds_ori
    return base_ds

data_name_dict = {
    'infer': 'answer_origin',
    'judge': 'answer_judge',
    'judge_vote': 'answer_judge_vote',
    'vote': 'vote',
    }

def get_latest_date(config, new_data_name, save_file):
    latest_paths = sorted([i for i in glob.glob(os.path.join(config['base_data_path'], new_data_name, '*')) if str(save_file) not in i])
    if len(latest_paths) == 0:
        latest_jsons = []
    else:
        latest_jsons = glob.glob(os.path.join(latest_paths[-1], '*'))
    latest_ds = []
    for tmp_path in latest_jsons:
        latest_ds.extend([json.loads(i) for i in open(tmp_path).readlines()])
    return latest_ds


def postprecess_func_vote(data):
    new_data = {**data['data_extra']['info']}
    new_data['vote'] = get_vote_result(data['answer'])
    return new_data

def postprecess_func_judge(data):
    new_data = {**data['data_extra']}
    try:
        tmp = data['answer'].split('final<|message|>')[-1]
        if 'true' in tmp:
            judge_result = True
        else:
            judge_result = False
    except:
        judge_result = None
    new_data['judge'] = judge_result
    return new_data

def postprecess_func_infer(data):
    try:
        new_data = {
            'info': data['data_extra'],
            'index': data['data_extra']['index'],
            'model_input': data['prompt'],
            'model_output': data['output']['choices'][0]['text'],
            'prompt': data['data_extra']['prompt'],
            'answer':  data['output']['choices'][0]['text'][len(data['prompt']):],
        }
    except:
        new_data = {
            'info': data['data_extra'],
            'index': data['data_extra']['index'],
            'model_output': None,
            'model_input': data['prompt'],
            'prompt': data['data_extra']['prompt'],
            'answer':  None,
        }
    return new_data

def get_vllm_data_postprocess_func(config, mode):
    if mode == 'infer':
        return postprecess_func_infer
    if mode in ['judge', 'judge_vote']:
        return postprecess_func_judge
    if mode == 'vote':
        return postprecess_func_vote
    return None

def vllm_data_preprocess(config, data, mode):
        if mode == 'infer':
            prompt = infer_prompt(data)
        elif mode == 'judge':
            prompt = judge_prompt(data)
        elif mode == 'judge_vote':
            prompt = judge_prompt(data, vote=True)
        elif mode == 'vote':
            prompt = vote_prompt(data)
        reasoning_level = config.get('reasoning_level', 'medium') 

        text = f'''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-12

reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|><|end|><|start|>user<|message|>
{prompt}
<|end|><|start|>assistant'''

        vllm_data = {
            "prompt": text,
            "echo": True,
            "seed": (lambda s: int(s) if s.isdigit() else 0)(data['index'].split('_')[-1]),
            "n": 1,
            "stream": False,
            "temperature": 0.8,
            "skip_special_tokens": False,
            'stop': ['<|endoftext|>', '<|return|>'],
            "max_tokens": config.get('max_tokens', 32000),
            'index': data['index'],
            'data_extra': data,
        }
        return vllm_data
    

@click.command()
@click.option('-c', '--config-path', default='config/oss.yaml', help='Path to the configuration YAML file.')
@click.option('-m', '--mode', default=None)
@click.option('-l', '--local_rank', default=0)
@click.option('-g', '--global_size', default=1)
@click.option('-s', '--save_file', default=None)
def main(config_path, local_rank, global_size, save_file, mode):

    ## update config
    config = yaml.safe_load(open(config_path))
    base_data_path = config['base_data_path']
    if mode is None:
        mode = config.get('mode', 'infer')
    else:
        config['mode'] = mode
    num_answer = config.get('num_answer', 8)
    logger.info(config)

    ## save data path
    if save_file is None:
        save_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_data_name = config.get('new_data_name', data_name_dict[mode])
    save_path = os.path.join(base_data_path, new_data_name, save_file, f'{local_rank}_{global_size}.jsonl')
    os.makedirs(os.path.join(base_data_path, new_data_name, save_file), exist_ok=True)
    logger.info(f'[save data to]: {save_path}')
    

    ## load prompt data
    base_ds = get_prompt_data(config, mode, num_answer)


    ## get latest data
    latest_ds = get_latest_date(config, new_data_name, save_file)

    ## prepare vllm input & save finished date
    # vllm input
    vllm_input_datas = []
    # finished data & info
    finished_data_info = {i["index"]: i for i in latest_ds if 'index' in i}
    finished = [] 

    count = 0
    for c, data in tqdm.tqdm(enumerate(base_ds)):
        key = data["index"]
        if key in finished_data_info and finished_data_info[key].get('answer', None) is not None:
            finished.append(finished_data_info[key])
            continue
        vllm_data = vllm_data_preprocess(config, data, mode)
        if count % global_size == local_rank:
            vllm_input_datas.append(vllm_data)
        count += 1

    logger.info(f'[total num]: {len(base_ds)}\n[latest num]: {len(latest_ds)}\n[finish num]: {len(finished)}\n')
    if local_rank == 0:
        logger.info(f'save data: {save_path}')
        with jsonlines.open(save_path, mode='w') as writer:
            writer.write_all(finished)


    ## vllm infer & postprecess
    postprocess = get_vllm_data_postprocess_func(config, mode)

    f = vllm_infer(config_path, chat=False, save_path=save_path, postprocess=postprocess)
    out = f.get_answer(vllm_input_datas)


    logger.info(f'save data: {save_path}')

if __name__ == '__main__':
    main()
