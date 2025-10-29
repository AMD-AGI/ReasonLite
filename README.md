# ReasonLite

## Data Generation Pipeline 

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

## Training
WIP


## Evaluate checkpoints 

Evaluate distilled checkpoints on AIME24 benchmark

```
cd eval
python eval_ckpts.py -c config/example.yaml
```



