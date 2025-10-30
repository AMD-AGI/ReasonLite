# ReasonLite

## Data Generation Pipeline 

### Setup environment

```
pip install -r requirements.txt
```

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

**Judge the correctness of results using pseudo-labels**

```
python3 infer.py -c config/oss.yaml -m judge_vote
```

### Filtering and converting to training format

**Omit incorrect solutions, and convert correct records to training-ready format.** You will need to specify the path to judged data

```
python3 utils/saving_to_training_format.py -d path/to/judged/data.jsonl
```

## Data formats

This repo reads a simple JSONL input and produces several JSONL outputs under a timestamped folder. Below is the on-disk layout and minimal examples.

### Folder layout

```
datas/<experiment>/
	info.jsonl                        # input prompts
	answer_origin/<timestamp>/0_1.jsonl      # raw generations
	answer_judge/<timestamp>/0_1.jsonl       # generations + correctness flag
	vote/<timestamp>/0_1.jsonl               # aggregated votes by prompt
	answer_judge_vote/<timestamp>/0_1.jsonl  # judged generations with embedded votes
```

Notes
- Files under output folders are sharded JSONL (e.g., 0_1.jsonl). The timestamp comes from the run.
- The field `index` may carry a per-sample suffix (e.g., `pol:2_3`) indicating a variant/generated attempt; the base ID (e.g., `pol:2`) refers to the original prompt.
- `model_input`/`model_output` preserve the full chat trace and can be long.

### Input: `info.jsonl`

One problem per line. `prompt` and `expected_answer` are menditory, while others are optional

```jsonl
{"prompt": "The front tires of a car wear out after 25,000 km, and the rear tires wear out after 15,000 km. When should the tires be swapped so that they wear out at the same time?", "expected_answer": "9375", "difficulty": "7/8", "index": "pol:0"}
```

### Output: `answer_origin/<timestamp>/0_1.jsonl`

Raw model generations with trace:

```jsonl
{"info": {"prompt": "...", "expected_answer": "9375", "difficulty": "7/8", "index": "pol:0_3"},
 "index": "pol:0_3",
 "model_input": "<full chat input ...>",
 "model_output": "<full chat output ...>",
 "prompt": "The front tires of a car wear out after 25,000 km, ...",
 "answer": "Swap after 9,375 km."}
```

### Output: `answer_judge/<timestamp>/0_1.jsonl`

Same as `answer_origin` plus a boolean `judge` field indicating correctness:

```jsonl
{"info": {"prompt": "...", "expected_answer": "392", "difficulty": "6/8", "index": "pol:2_3"},
 "index": "pol:2_3",
 "model_input": "<...>",
 "model_output": "<...>",
 "prompt": "The largest three-digit number ...",
 "answer": "393",
 "judge": false}
```

### Output: `vote/<timestamp>/0_1.jsonl`

Per-prompt aggregated votes (no trace):

```jsonl
{"prompt": "The largest three-digit number ...",
 "expected_answer": "392",
 "difficulty": "6/8",
 "index": "pol:2",
 "vote": {"393": 2, "392": 2}}
```

### Output: `answer_judge_vote/<timestamp>/0_1.jsonl`

Judged generations with embedded upstream vote tallies and label source:

```jsonl
{"info": {"prompt": "The largest three-digit number ...",
					 "expected_answer": "392",
					 "difficulty": "6/8",
					 "index": "pol:2",
					 "vote": {"393": 2, "392": 2},
					 "label_source": "dataset"},
 "index": "pol:2_3",
 "model_input": "<...>",
 "model_output": "<...>",
 "prompt": "The largest three-digit number ...",
 "answer": "393",
 "judge": false}
```

## Training
WIP


## Evaluate checkpoints 

See the following example for evaluating distilled checkpoints on AIME24 benchmark

```
cd eval
export CUDA_VISIBLE_DEVICES=0

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model <path_to_check_point> \
    --chat_template_name default \
    --system_prompt_name disabled \
    --output_dir results \
    --tensor_parallel_size 1 \
    --data_id zwhe99/aime90 \
    --bf16 True \
    --split 2024 \
    --max_model_len 32768 \
    --temperature 0.8 \
    --top_p 1.0 \
    --n 16

```



