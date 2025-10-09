import json
import jsonlines
import os
import numpy as np

data_paths = [
    'datas/example/answer_judge_vote/20251009_083232/0_1.jsonl',
]

data = []
for data_path in data_paths:
    data.extend([json.loads(i) for i in open(data_path).readlines()])

save_path = data_path.split('/answer_judge')[0] + '/format_data'
os.makedirs(save_path, exist_ok=True)

print(f'[load]: {data_path}\n[save]: {save_path}')

out = []
acc = []
for i in data:
    judge = i['judge']
    if judge is None:
        acc.append(-1)
        continue
    elif judge is False:
        acc.append(0)
        continue
    model_output = i['model_output']
    if len(model_output.split('<|start|>assistant<|channel|>')) != 3:
        acc.append(-2)
        continue
    acc.append(1)
    answer = model_output.split('<|end|><|start|>assistant<|channel|>analysis<|message|>')[1]
    answer = "<think>\n" + answer.replace('<|end|><|start|>assistant<|channel|>final<|message|>',  '\n</think>')
    prompt = i['model_input'].split('<|start|>user<|message|>')[1].split('<|end|>')[0].strip()
    out.append({"messages": [{"role": "user", "content": prompt},  {"role": "assistant", "content": answer}]})

print(f'[total data]: {len(data)}')
print(f'[keeped data]: {len(out)}')
print(f'[format mismatch ratio]: {float((np.array(acc) == -2).mean())}')
print(f'[error ratio]: {float((np.array(acc) == -1).mean())}')
print(f'[wrong ratio]: {float((np.array(acc) == 0).mean())}')
print(f'[correct ratio]: {float((np.array(acc) == 1).mean())}')


chunk_size = 100000
num_chunks = (len(out) + chunk_size - 1) // chunk_size
np.random.shuffle(out)

for i in range(num_chunks):
    chunk = out[i*chunk_size:(i+1)*chunk_size]
    save_path_tmp = os.path.join(save_path, f"data_part{i+1}.jsonl")
    print(f"Saving {len(chunk)} records to {save_path}")
    with jsonlines.open(save_path_tmp, mode='w') as writer:
        writer.write_all(chunk)

