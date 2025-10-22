import os
import glob
import json 
from enum import Enum
from tqdm import tqdm
from copy import deepcopy

from utils.tools import get_last_boxed
from utils.tools import get_last_boxed, get_vote_result
from utils.prompt_func import infer_prompt, vote_prompt, judge_prompt
from utils.logging_utils import logger


class Mode(str, Enum):
    """Enumeration for different reasoning modes."""
    INFER = 'infer'
    JUDGE = 'judge'
    JUDGE_VOTE = 'judge_vote'
    VOTE = 'vote'
    
    @classmethod
    def from_str(cls, mode_str: str):
        for mode in cls:
            if mode.value == mode_str:
                return mode
        raise ValueError(f"Unknown ReasonMode: {mode_str}")
    
    
class ReasoningModeHandler:
    """Unified handler for mode, preprocessing, postprocessing, and data IO."""

    data_name_map = {
        Mode.INFER: 'answer_origin',
        Mode.JUDGE: 'answer_judge',
        Mode.JUDGE_VOTE: 'answer_judge_vote',
        Mode.VOTE: 'vote',
    }

    def __init__(self, config: dict, mode: Mode):
        self.config = config
        self.mode = mode
        self.base_data_path = config['base_data_path']

    @property
    def data_name(self) -> str:
        return self.data_name_map[self.mode]

    # ------------- Data loading helpers -------------
    def get_prompt_data(self, num_answer: int):
        base_data_path = self.base_data_path
        if self.mode == Mode.INFER:
            prompt_json_paths = [os.path.join(base_data_path, self.config.get('data_info_name', 'info.jsonl'))]
        else:
            prompt_path = os.path.join(base_data_path, self.data_name_map[Mode.INFER])
            prompt_path = sorted(glob.glob(os.path.join(prompt_path, '*')))[-1]
            prompt_json_paths = glob.glob(os.path.join(prompt_path, '*.jsonl'))
        logger.info(f'[load data from]: {prompt_json_paths}')

        base_ds_ori = []
        for prompt_json_paths_tmp in prompt_json_paths:
            base_ds_ori.extend([json.loads(i) for i in open(prompt_json_paths_tmp).readlines()])

        if self.mode == Mode.INFER:
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
        elif self.mode == Mode.VOTE:
            data_by_id = {}
            for i in tqdm(base_ds_ori):
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
        elif self.mode == Mode.JUDGE_VOTE:
            vote_info_path = sorted(glob.glob(os.path.join(base_data_path, self.data_name_map[Mode.VOTE], '*', '*.jsonl')))[-1]
            vote_info = [json.loads(i) for i in open(vote_info_path).readlines()]
            vote_info_dict = {i['index']: i for i in vote_info}
            base_ds = []
            for data in base_ds_ori:
                data['info'] = vote_info_dict[data['index'].split('_')[0]]
                base_ds.append(data)
        else:
            base_ds = base_ds_ori
        return base_ds

    def get_latest_data(self, save_file: str):
        new_data_name = self.data_name
        latest_paths = sorted([
            i for i in glob.glob(os.path.join(self.base_data_path, new_data_name, '*'))
            if str(save_file) not in i
        ])
        if len(latest_paths) == 0:
            latest_jsons = []
        else:
            latest_jsons = glob.glob(os.path.join(latest_paths[-1], '*'))
        latest_ds = []
        for tmp_path in latest_jsons:
            latest_ds.extend([json.loads(i) for i in open(tmp_path).readlines()])
        return latest_ds

    # ------------- Pre/Post-process -------------
    def preprocess(self, data: dict) -> dict:
        mode = self.mode
        if mode == Mode.INFER:
            prompt = infer_prompt(data)
        elif mode == Mode.JUDGE:
            prompt = judge_prompt(data)
        elif mode == Mode.JUDGE_VOTE:
            prompt = judge_prompt(data, vote=True)
        elif mode == Mode.VOTE:
            prompt = vote_prompt(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        reasoning_level = self.config.get('reasoning_level', 'medium')
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
            "max_tokens": self.config.get('max_tokens', 32000),
            'index': data['index'],
            'data_extra': data,
        }
        return vllm_data

    @property
    def postprocess(self):
        mode = self.mode

        def process_vote(data):
            new_data = {**data['data_extra']['info']}
            new_data['vote'] = get_vote_result(data['answer'])
            return new_data

        def process_judge(data):
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

        def process_infer(data):
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

        if mode == Mode.INFER:
            return process_infer
        if mode in [Mode.JUDGE, Mode.JUDGE_VOTE]:
            return process_judge
        if mode == Mode.VOTE:
            return process_vote
        return None
    
