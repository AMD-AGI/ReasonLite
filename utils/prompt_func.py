from utils.tools import get_last_boxed

def infer_prompt(i):
    prompt = i['prompt'] + '\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return prompt

def vote_prompt(data):
    answers = ''
    for c, i in enumerate(data['answer']):
        answers += f'Answer Index {c}:\n{i}\n\n'
    prompt = f'''I have a math problem, and a large language model has generated n answers. Please help me summarize how many times each unique answer appears. If two answers are very similar, you can group them together. Here are the n answers:

{answers}
...

Please output the results strictly in JSON, with the final answer enclosed like this:

\\boxed{{"answer1": count1, "answer2": count2, ...}}

For example, if the input answers are [12, x=25, 25, 25, 12, -2], the output should be:

\\boxed{{"25": 3, "12": 2, "-2": 1}}
'''
    return prompt


def judge_prompt(data, vote=False):
    model_answer_box = get_last_boxed(data['model_output'])
    if not vote:
        target_answer = data['info']['expected_answer']
    else:
        data['info']['label_source'] = 'dataset'
        try:
            target_answer = (list(data['info']['vote'].keys()) + ['none'])[0]
            vote_ratio = 0.9
            min_vote = 4
            if target_answer != 'none' and list(data['info']['vote'].values())[0] / sum(list(data['info']['vote'].values())) >= vote_ratio and  list(data['info']['vote'].values())[0] >= min_vote:
                target_answer = (list(data['info']['vote'].keys()) + ['none'])[0]
                data['info']['label_source'] = 'vote'
            else:
                target_answer = data['info']['expected_answer']
        except:
            target_answer = data['info']['expected_answer']

    prompt = f'''You are a math reasoning assistant. Your task is to check if the **model's answer** matches the **ground truth answer**.  

- Consider mathematical equivalence, not just string matching.  
- Simplify or transform expressions if needed.  
- Only return the result in the required JSON format.

Here is the question and answers:

Ground truth answer:
{target_answer}

Model's answer:
{model_answer_box}

Please analyze step by step and determine if the two answers are mathematically the same. Output strictly in JSON, put your final answer within \\boxed{{"match": true or false}}.
'''
    return prompt


