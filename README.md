# ReasonLite

## Start Vllm Server

```
python3 vllm_start.py -c config/oss.yaml
```

## Synthetic Data

**Generate data answers**

```
python3 infer.py -c config/oss.yaml -m infer
```

**Judge the correctness of answer**


```
python3 infer.py -c config/oss.yaml -m judge
```

## Pseudo Label

**Obtain pseudo-labels through voting**

```
python3 infer.py -c config/oss.yaml -m vote
```

**Judge the correctness of results using pseudo-labels**

```
python3 infer.py -c config/oss.yaml -m judge_vote
```

