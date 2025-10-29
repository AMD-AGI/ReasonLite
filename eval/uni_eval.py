import os
import re
import fire
import math
import torch
import sympy as sp

from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, ClassLabel

from vllm import LLM, SamplingParams
from math_verify import verify, parse

from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex

from utils.openmathinst_utils import process_results
from utils.polymath.judge import pm_judge
from utils.data_utils import write_jsonl, write_json, read_jsonl
from utils.chat_template import CHAT_TEMPLATE, SYSTEM_PROMPT, PREFIX_PROMPT, SUFFIX_PROMPT

DATASET_INFO = {
    "zwhe99/MATH": {
        "default_split": "math500",
        "problem_key": "problem",
        "answer_key": "expected_answer",
        "category_keys": ["level", "type"]
    },
    "zwhe99/aime90": {
        "default_split": "2024",
        "problem_key": "problem",
        "answer_key": "expected_answer",
    },
    "zwhe99/amc23": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "answer",
    },
    "zwhe99/simplerl-minerva-math": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "math-ai/aime25": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "zwhe99/simplerl-OlympiadBench": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "final_answer",
    },
    "zwhe99/gpqa_diamond_mc": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
        "category_keys": ["domain"]
    },
    "zwhe99/pm-en": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "answer",
        "category_keys": ["level"]
    }
}

class OBJudge:
    def __init__(self):
        # Map of special symbols to their replacements
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8  # Default precision for comparison

    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())   
        
        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        # Translates plus-minus signs into separate expressions
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        
        return new_expr_list
    
    def judge(self, expression1, expression2, precision=1e-8):
        # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
        # Default precision is a list for supporting multiple expressions
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            # print("Exactly equal")
            return True
        
        # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)
        
        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Set up a list for allowed errors
        if len(precision) <= 1:
            precision = precision * len(temp_list1)
        
        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If no match was found, return False
                return False

        # If all elements are matched, return True
        return True
    
    def is_interval(self, expr):
        # Checks if an expression is an interval
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        # Replaces the symbol for pi in sympy expressions with its numerical value
        return expression_sympy.subs(self.pi, math.pi)
    
    def is_equal(self, expression1, expression2):
        # Default first expression is ground truth. Check if expressions are equal in different aspects
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("Equivalent natively")
            return True

        # First check if both are intervals
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("Interval equivalent")
                    return True
            except:
                return False

        # Then check for numerical equality
        try:
            if self.numerical_equal(expression1, expression2):
                # print("Numerically equivalent")
                return True
        except:
            pass
        
        # Then check if expressions are mathematically equal
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                # print("Expression equivalent")
                return True
        except:
            pass
            
        # Lastly, check for equation equality
        try:
            if self.equation_equal(expression1, expression2):
                # print("Equation equivalent")
                return True
        except:
            pass
            
        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        # Check if two numerical values are equal within an allowed error range
        # Includes possible percentage cases
        reference = float(expression1)
        prediction = float(expression2)
        
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        
        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False
    

    def expression_equal(self, exp1, exp2):
        # Check if two expressions are mathematically equivalent
        # Extract expression and use sympy for equivalence checking
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()
        
        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(f"These two numbers cannot be calculated by the current computer for: \"{str(expr1_sym)}\" and \"{str(expr2_sym)}\"")
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()
                    
                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        # Check if two equations are mathematically equivalent
        # Simplify equations and use sympy for equivalence checking
        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')

            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            equation = Eq(lhs_expr, rhs_expr)

            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # Check if two intervals are mathematically equivalent
        def compare_two_interval(inter1, inter2):
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False
            
            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True
            
        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")
            
            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        # Preprocess expressions to extract and replace special symbols
        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str
                
            return results
        
        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]
            
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression
        
        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2
    
    def can_compute_power(self, expr):
        # Checks if a power expression can be computed
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000  # Adjust based on computing environment
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True  # Not a power expression, can compute

def pass_at_k(correct_lst: list[bool], k: int) -> float:
    assert k > 0, "k must be greater than 0"
    assert k <= len(correct_lst), "k must be less than or equal to the length of `correct_lst`"

    num_samples = len(correct_lst)
    num_correct = sum(correct_lst)
    if num_correct == 0:
        return 0.0
    elif (num_samples - num_correct) < k:
        return 1.0
    else:
        log_ratio = 0.0
        for i in range(k):
            log_ratio += math.log(num_samples - num_correct - i) - math.log(num_samples - i)
        return 1.0 - math.exp(log_ratio)

def bulid_choice_prompt(question: str, choices: list[str]):
    prompt = f"{question}\n\n\n"
    options = [chr(65 + i) for i in range(len(choices))]
    for option, choice in zip(options, choices):
        prompt += f"({option}) {choice}\n"

    prompt += "\nPlease write your final answer in the form of "
    for oid, opt in enumerate(options):
        if oid != len(options) - 1:
            prompt += f"\\boxed{{{opt}}}, "
        else:
            prompt += f"or \\boxed{{{opt}}}"
    return prompt

def eval(
    # required
    base_model: str = None,
    chat_template_name: str = "default",
    system_prompt_name: str = "disabled",
    prefix_prompt_name: str = "disabled",
    suffix_prompt_name: str = "disabled",
    output_dir: str = None,

    # model
    bf16: bool = False,
    fp16: bool = False,
    tensor_parallel_size: int = 8,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.95,

    # data
    data_dir: str = None, # If provided, the data will loaded from data_dir/data_id
    data_id: str = None,
    split: str = None,
    subset: str = None,
    start_idx: int = None,
    end_idx: int = None,

    # gen
    max_model_len: int = 32768,
    temperature: float = 0.6,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    n: int = 1,

    seed: int = 42,
):
    # Path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generation_file = os.path.join(output_dir, "generation.jsonl")
    result_file = os.path.join(output_dir, "result.log")
    config_file = os.path.join(output_dir, "config.json")

    # Sanity check
    assert not (start_idx is not None and end_idx is None or start_idx is None and end_idx is not None), "start_idx and end_idx must be provided together"
    if start_idx is not None and end_idx is not None:
        assert end_idx > start_idx, "end_idx must be greater than start_idx"

    if isinstance(split, int):
        split = str(split)

    # save config
    write_json(config_file, locals())

    # Get dataset info
    problem_key = DATASET_INFO[data_id]["problem_key"]
    answer_key = DATASET_INFO[data_id]["answer_key"]
    choice_key = DATASET_INFO[data_id]["choice_key"] if "choice_key" in DATASET_INFO[data_id] else None

    # load model
    llm = LLM(
        model=base_model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32),
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if chat_template_name is not None and chat_template_name != "default":
        tokenizer.chat_template = CHAT_TEMPLATE[chat_template_name]

    # Load data
    if subset is None and "default_subset" in DATASET_INFO[data_id]:
        subset = DATASET_INFO[data_id]["default_subset"]

    if data_dir is None:
        if subset is None:
            test_dataset = load_dataset(data_id)
        else:
            test_dataset = load_dataset(data_id, subset)
    else:
        if subset is None:
            test_dataset = load_from_disk(os.path.join(data_dir, data_id))
        else:
            test_dataset = load_from_disk(os.path.join(data_dir, data_id, subset))

    if split is None:
        split = DATASET_INFO[data_id]["default_split"]

    test_dataset = test_dataset[split]

    if start_idx is not None and end_idx is not None:
        test_dataset = test_dataset.select(range(start_idx, end_idx))

    system_message = []
    if system_prompt_name != "disabled":
        system_message = [{"role": "system", "content": SYSTEM_PROMPT[system_prompt_name]}]

    prefix_prompt = ""
    if prefix_prompt_name != "disabled":
        prefix_prompt = PREFIX_PROMPT[prefix_prompt_name]

    suffix_prompt = ""
    if suffix_prompt_name != "disabled":
        suffix_prompt = SUFFIX_PROMPT[suffix_prompt_name]

    prompts = [
        tokenizer.apply_chat_template(
            conversation=system_message + [
                {
                    "role": "user",
                    "content": prefix_prompt + td[problem_key] if not choice_key else bulid_choice_prompt(td[problem_key], td[choice_key]) + suffix_prompt
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for td in test_dataset
    ]

    prompt_lens = [
        len(tokenizer.apply_chat_template(
            conversation=system_message + [
                {
                    "role": "user",
                    "content": prefix_prompt + td[problem_key] if not choice_key else bulid_choice_prompt(td[problem_key], td[choice_key]) + suffix_prompt
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        ))
        for td in test_dataset
    ]

    # repeat n times
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_model_len,
            n=1,
            seed=seed + i,
        ) for p, pl in zip(prompts, prompt_lens) for i in range(n)
    ]
    prompts = [p for p in prompts for i in range(n)]
    prompt_lens = [p for p in prompt_lens for i in range(n)]

    # generate
    if os.path.exists(generation_file) and os.path.getsize(generation_file) > 0:
        print(f"Loading generations from {generation_file}")
        generations = read_jsonl(generation_file)
    else:
        generations = []
        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == len(prompts)

        for tdi, td in enumerate(test_dataset):
            local_outputs = outputs[tdi * n: (tdi + 1) * n]
            local_prompts = prompts[tdi * n: (tdi + 1) * n]
            local_prompt_lens = prompt_lens[tdi * n: (tdi + 1) * n]

            new_td = deepcopy(td)
            new_td["prompt"] = local_prompts[0]
            new_td["prompt_length"] = local_prompt_lens[0]
            new_td["response"] = [lo.outputs[0].text for lo in local_outputs]
            new_td["response_length"] = [len(lo.outputs[0].token_ids) for lo in local_outputs]
            new_td["finish_reason"] = [lo.outputs[0].finish_reason for lo in local_outputs]
            generations.append(new_td)

        write_jsonl(generation_file, generations)

    # compute correctness and pass@k (sample-level)
    ks = [2 ** e for e in range(0, 7)]
    ks = [k for k in ks if (2 * k) <= n or k == 1]
    for g in tqdm(generations, desc="computing correctness", total=len(generations)):
        gt_answer = g[answer_key]
        if isinstance(test_dataset.features[answer_key], ClassLabel):
            gt_answer = test_dataset.features[answer_key].int2str(gt_answer)
        else:
            if isinstance(gt_answer, list):
                assert len(gt_answer) == 1, "gt_answer must be a single string"
                gt_answer = str(gt_answer[0])
            else:
                gt_answer = str(gt_answer)

        if data_id == "zwhe99/simplerl-OlympiadBench":
            # Note: Olympiadbench has its offical judge which do not support duplicated `boxed` in the response.
            # Therefore, we strip the `reasoning` part in the response if it exists.
            scorer = OBJudge()
            g["correct"] = [
                (
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=True,
                    ) or
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    ) or
                    verify(parse(f"\\boxed{{${gt_answer}}}$"), parse(resp))
                    or scorer.judge(gt_answer, resp if "</think>" not in resp else resp.split("</think>")[1].strip(), 1e-8)
                ) for resp in g["response"]
            ]
        elif data_id == "zwhe99/pm-en":
            g["correct"] = [
                (
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=True,
                    ) or
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    ) or
                    verify(parse(f"\\boxed{{${gt_answer}}}$"), parse(resp))
                    or pm_judge(resp, gt_answer)
                ) for resp in g["response"]
            ]
        else:
            g["correct"] = [
                (
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=True,
                    ) or
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    ) or
                    verify(parse(f"\\boxed{{${gt_answer}}}$"), parse(resp))
                ) for resp in g["response"]
            ]

        for k in ks:
            g[f"pass@{k}"] = pass_at_k(g["correct"], k)
    write_jsonl(generation_file, generations)

    # dataset-level metrics
    with open(result_file, "w") as f:
        for k in ks:
            f.write(f"pass@{k} >>>\n")
            if "category_keys" in DATASET_INFO[data_id] and len(DATASET_INFO[data_id]["category_keys"]) > 0:
                for ck in DATASET_INFO[data_id]["category_keys"]:
                    all_cate = sorted(list(set([g[ck] for g in generations])))
                    for cate in all_cate:
                        pass_prob_lst = [g[f"pass@{k}"] for g in generations if g[ck] == cate]
                        pass_prob_avg = sum(pass_prob_lst) / len(pass_prob_lst)
                        f.write(f"{cate}: {pass_prob_avg * 100:.1f}\n")

            # overall
            pass_prob_lst = [g[f"pass@{k}"] for g in generations]
            pass_prob_avg = sum(pass_prob_lst) / len(pass_prob_lst)
            f.write(f"Overall: {pass_prob_avg * 100:.1f}\n\n")

    # print the result file
    with open(result_file, "r") as f:
        print(f.read())

if __name__ == "__main__":
    fire.Fire(eval)
