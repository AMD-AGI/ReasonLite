import os
import glob
import json
from tqdm import tqdm
from copy import deepcopy

from utils.tools import get_last_boxed, get_vote_result
from utils.prompt_func import infer_prompt, vote_prompt, judge_prompt
from utils.logging_utils import logger

    
    
class BaseReasoningModeHandler:
    """Base handler defining common IO and request-building utilities."""

    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.base_data_path = config['base_data_path']

    @property
    def data_name(self) -> str:
        raise NotImplementedError

    # helper functions
    def _latest_origin_jsonls(self):
        """Return list of jsonl paths from the latest 'answer_origin' folder."""
        origin_root = os.path.join(self.base_data_path, 'answer_origin')
        latest_origin_dir = sorted(glob.glob(os.path.join(origin_root, '*')))
        if not latest_origin_dir:
            return []
        return glob.glob(os.path.join(latest_origin_dir[-1], '*.jsonl'))

    def _read_jsonl_many(self, paths):
        out = []
        for p in paths:
            out.extend([json.loads(i) for i in open(p).readlines()])
        return out

    # data loading
    def get_prompt_data(self, num_answer: int):
        raise NotImplementedError

    def get_latest_data(self, save_file: str):
        new_data_name = self.data_name
        latest_paths = sorted([
            i for i in glob.glob(os.path.join(self.base_data_path, new_data_name, '*'))
            if str(save_file) not in i
        ])
        latest_jsons = glob.glob(os.path.join(latest_paths[-1], '*')) if latest_paths else []
        latest_ds = []
        for tmp_path in latest_jsons:
            latest_ds.extend([json.loads(i) for i in open(tmp_path).readlines()])
        return latest_ds

    # build request
    def _wrap_prompt(self, prompt: str) -> str:
        reasoning_level = self.config.get('reasoning_level', 'medium')
        return (
            f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-12

reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|><|end|><|start|>user<|message|>
{prompt}
<|end|><|start|>assistant"""
        )

    def preprocess(self, data: dict, temperature: float, top_p: float) -> dict:
        """Subclass should prepare the user-visible prompt. This wraps and builds vLLM input."""
        user_prompt = self._build_user_prompt(data)
        text = self._wrap_prompt(user_prompt)
        vllm_data = {
            'prompt': text,
            'echo': True,
            'seed': (lambda s: int(s) if s.isdigit() else 0)(data['index'].split('_')[-1]),
            'n': 1,
            'stream': False,
            'temperature': temperature,
            'top_p': top_p,
            'skip_special_tokens': False,
            'stop': ['<|endoftext|>', '<|return|>'],
            'max_tokens': self.config.get('max_tokens', 32000),
            'index': data['index'],
            'data_extra': data,
        }
        return vllm_data

    def _build_user_prompt(self, data: dict) -> str:
        raise NotImplementedError

    @property
    def postprocess(self):
        raise NotImplementedError


class InferModeHandler(BaseReasoningModeHandler):
    @property
    def data_name(self) -> str:
        return 'answer_origin'

    def get_prompt_data(self, num_answer: int):
        base_data_path = self.base_data_path
        prompt_json_paths = [os.path.join(base_data_path, self.config.get('data_info_name', 'info.jsonl'))]
        logger.info(f'[load data from]: {prompt_json_paths}')
        base_ds_ori = self._read_jsonl_many(prompt_json_paths)

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
        return base_ds

    def _build_user_prompt(self, data: dict) -> str:
        return infer_prompt(data)

    @property
    def postprocess(self):
        def process_infer(data):
            try:
                new_data = {
                    'info': data['data_extra'],
                    'index': data['data_extra']['index'],
                    'model_input': data['prompt'],
                    'model_output': data['output']['choices'][0]['text'],
                    'prompt': data['data_extra']['prompt'],
                    'answer': data['output']['choices'][0]['text'][len(data['prompt']):],
                }
            except Exception:
                new_data = {
                    'info': data['data_extra'],
                    'index': data['data_extra']['index'],
                    'model_output': None,
                    'model_input': data['prompt'],
                    'prompt': data['data_extra']['prompt'],
                    'answer': None,
                }
            return new_data

        return process_infer


class JudgeModeHandler(BaseReasoningModeHandler):
    @property
    def data_name(self) -> str:
        return 'answer_judge'

    def get_prompt_data(self, num_answer: int):
        prompt_json_paths = self._latest_origin_jsonls()
        logger.info(f'[load data from]: {prompt_json_paths}')
        return self._read_jsonl_many(prompt_json_paths)

    def _build_user_prompt(self, data: dict) -> str:
        return judge_prompt(data)

    @property
    def postprocess(self):
        def process_judge(data):
            new_data = {**data['data_extra']}
            try:
                tmp = data['answer'].split('final<|message|>')[-1]
                judge_result = True if 'true' in tmp else False
            except Exception:
                judge_result = None
            new_data['judge'] = judge_result
            return new_data

        return process_judge


class JudgeVoteModeHandler(BaseReasoningModeHandler):
    @property
    def data_name(self) -> str:
        return 'answer_judge_vote'

    def get_prompt_data(self, num_answer: int): 
        base_data_path = self.base_data_path
        prompt_json_paths = self._latest_origin_jsonls()
        logger.info(f'[load data from]: {prompt_json_paths}')
        base_ds_ori = self._read_jsonl_many(prompt_json_paths)

        vote_info_path = sorted(
            glob.glob(os.path.join(base_data_path, 'vote', '*', '*.jsonl'))
        )[-1]
        vote_info = [json.loads(i) for i in open(vote_info_path).readlines()]
        vote_info_dict = {i['index']: i for i in vote_info}
        base_ds = []
        for data in base_ds_ori:
            data['info'] = vote_info_dict[data['index'].split('_')[0]]
            base_ds.append(data)
        return base_ds

    def _build_user_prompt(self, data: dict) -> str:
        return judge_prompt(data, vote=True)

    @property
    def postprocess(self):
        def process_judge(data):
            new_data = {**data['data_extra']}
            try:
                tmp = data['answer'].split('final<|message|>')[-1]
                judge_result = True if 'true' in tmp else False
            except Exception:
                judge_result = None
            new_data['judge'] = judge_result
            return new_data

        return process_judge


class VoteModeHandler(BaseReasoningModeHandler):
    @property
    def data_name(self) -> str:
        return 'vote'

    def get_prompt_data(self, num_answer: int):
        prompt_json_paths = self._latest_origin_jsonls()
        logger.info(f'[load data from]: {prompt_json_paths}')
        base_ds_ori = self._read_jsonl_many(prompt_json_paths)

        data_by_id = {}
        for i in tqdm(base_ds_ori):
            index = i['index']
            prompt_index = index.split('_')[0]
            i['info']['index'] = str(prompt_index)
            gt = i['info']['expected_answer']
            answer = get_last_boxed(i['model_output'])
            if prompt_index not in data_by_id:
                data_by_id[prompt_index] = []
            data_by_id[prompt_index].append({'gt': gt, 'answer': answer, 'info': i['info']})

        check_answer = []
        for prompt_index, items in data_by_id.items():
            answer = [it['answer'] for it in items]
            check_answer.append({
                'index': prompt_index,
                'answer': answer,
                'gt': items[0]['gt'],
                'info': items[0]['info'],
            })
        return check_answer

    def _build_user_prompt(self, data: dict) -> str:
        return vote_prompt(data)

    @property
    def postprocess(self):
        def process_vote(data):
            new_data = {**data['data_extra']['info']}
            new_data['vote'] = get_vote_result(data['answer'])
            return new_data

        return process_vote


HANDLER_MAP = {
    'infer': InferModeHandler,
    'judge': JudgeModeHandler,
    'judge_vote': JudgeVoteModeHandler,
    'vote': VoteModeHandler,
}

# Public list of available modes for CLI and validation convenience
AVAILABLE_MODES = sorted(HANDLER_MAP.keys())


class ReasoningModeHandler(BaseReasoningModeHandler):
    """
    Factory wrapper that returns a mode-specific handler instance.
    """

    def __new__(cls, config: dict, mode: str):
        mode_key = str(mode).lower()
        handler_cls = HANDLER_MAP.get(mode_key)
        if handler_cls is None:
            raise ValueError(f"Unknown mode: {mode}")
        instance = super().__new__(handler_cls)
        # Explicitly initialize the concrete handler since __init__ won't be
        # called automatically when returning an instance of a different class.
        handler_cls.__init__(instance, config, mode_key)
        return instance

    def __init__(self, config: dict, mode: str): 
        # No-op: actual init is performed in __new__ on the concrete subclass.
        pass
 
    
