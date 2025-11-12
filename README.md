# ReasonLite

## Setup environment

```
pip install -r requirements.txt
```

## Data Generation Pipeline 

You will need to edit the config `oss.yaml` file under `config/` directory or to create a new one based on your needs, then you will be able to run through the following pipeline.

### Start vllm Server

```
python3 vllm_start.py -c config/oss.yaml
```

### Synthetic Data

**Generate data answers**

```
python3 infer.py -c config/oss.yaml -m infer
```

**Judge the correctness of answer**


```
python3 infer.py -c config/oss.yaml -m judge
```

### Pseudo Label

**Obtain pseudo-labels through voting**

```
python3 infer.py -c config/oss.yaml -m vote
```

**Judge the correctness of results using pseudo-labels**. Running through `vote` is a pre-requisite for this step.

```
python3 infer.py -c config/oss.yaml -m judge_vote
```

### Filtering and converting to training format

**Omit incorrect solutions, and convert correct records to training-ready format.** You will need to specify the path to judged data, currently only support gpt-oss generated data.

```
python3 utils/saving_to_training_format.py -d path/to/judged/data.jsonl
```

## Data formats

This repo reads a simple JSONL input and produces several JSONL outputs under a timestamped folder. Below is the on-disk layout and minimal examples.

### Expriment Directory Layout

```
datas/
└── <experiment_name>/
    ├── info.jsonl           # input prompts
    ├── answer_origin/       # folder for raw generations
    │   └── <timestamp>/
    │       └── 0_1.jsonl
    ├── answer_judge/        # generations w/ correctness flag
    │   └── <timestamp>/
    │       └── 0_1.jsonl
    ├── vote/                # majority votes by prompt
    │   └── <timestamp>/
    │       └── 0_1.jsonl
    └── answer_judge_vote/   # judged generations with majority votes
        └── <timestamp>/
            └── 0_1.jsonl
```

Notes
- Files under output folders are sharded JSONL (e.g., 0_1.jsonl). The timestamp comes from the run.
- The field `index` may carry a per-sample suffix (e.g., `pol:2_3`) indicating a variant/generated attempt; the base ID (e.g., `pol:2`) refers to the original prompt.
- `model_input`/`model_output` preserve the full chat trace and can be long.

### JSONL examples

#### Input

The `info.jsonl` and `vote/<timestamp>/0_1.jsonl` take the following format:

```jsonl
{
    "prompt": "What is the results of 2 + 3?",
    "expected_answer": "5", 
    "index": "pol:0",
    "vote": {"5": 4, "392": 0} # specific to vote/<timestamp>/0_1.jsonl
}
```
The `prompt` field contains a math problem. And `expected_answer` store supposedly ground truth answer to the question for downstream LLM judgement. You will also need to specify a `index` for better tracking and resumming generation. And `vote` collects per-prompt aggregated votes.


#### Output

The rest jsonl file, `answer_origin/<timestamp>/0_1.jsonl`, `answer_judge/<timestamp>/0_1.jsonl`, and `answer_judge_vote/<timestamp>/0_1.jsonl` are structured like:

```jsonl
{
    "info": "<input info from `info.jsonl`>",
    "index": "pol:0_3",
    "model_input": "<full input ...>",
    "model_output": "<full output with input prepended...>",
    "prompt": "The front tires of a car wear out after 25,000 km, ...",
    "answer": "<model output ...>",
    "judge": true, # specific to answer_judge and answer_judge_vote
}
```
The jsonl files under `answer_origin` contain LLM-generated trajectoris, while `answer_judge` and `answer_judge_vote` adds a `judge` flag to identify wether LLM's solutions are correct based on different ground truth.


## Evaluate checkpoints 

See the following example for evaluating distilled checkpoints on AIME24 benchmark,

```bash
cd eval
export CUDA_VISIBLE_DEVICES=0
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model <path_to_your_checkpoint> \
    --chat_template_name default \
    --system_prompt_name disabled \
    --output_dir results \
    --tensor_parallel_size 1 \
    --data_id zwhe99/aime90 \
    --bf16 True \
    --split 2024 \
    --max_model_len 32768 \
    --temperature 0.0 \
    --top_p 1.0 \
    --n 16


```



