import json
import random
import numpy as np

from datasets import Dataset
from typing import List


def prepate_prompts(n: int,
                    template_src: str = '../data/templates.json',
                    titles_src: str = '../data/titles.txt') -> List[str]:
    
    """
    Prepare prompts by inserting a title into a template.

    Args:
        n: number of generated prompts
        template_src: path to the file with prompt templates
        titles_src: path to the file with movie titles
    
    Return:
        prompts: a list of prepared prompts
    """
    
    with open(template_src, 'r') as f:
        templates = json.load(f)

    with open(titles_src, 'r') as f:
        titles = f.read().split("\n")
    
    prompts = []
    chosen_titles = list(np.random.randint(0, len(titles), n))

    for title_id in chosen_titles:
    
        title = titles[title_id]
        template_n = random.randint(1, len(templates))

        instruction = templates[str(template_n)].format(input=title)

        prompts.append(instruction)
    
    return prompts


def prepare_dataset(prompts: List[str],
                    texts: List[str],
                    logits: List[float]) -> Dataset:
    """
    Prepare ranking of two texts of the same prompt.

    Args:
        prompts: list of initial prompts; len(prompts) == n
        texts: list of generated answers; len(texts) == 2n
        logits: list of rewards of generated answers; len(logits) == 2n
    
    Return:
        Dataset: dataset generated via ranking
    """

    """
    len(texts) == len(logits) == len(prompts)*2
    """
    pairs_text = zip(texts[::2], texts[1::2])
    pairs_logits = zip(logits[::2], logits[1::2])

    dpo_data = {'prompt': [], 'chosen': [], 'rejected': []}
    for i in range(len(prompts)):
        dpo_data['prompt'].append(prompts[i])
        if pairs_logits[i][0] > pairs_logits[i][1]:
            dpo_data['chosen'].append(pairs_text[i][0])
            dpo_data['rejected'].append(pairs_text[i][1])
        else:
            dpo_data['chosen'].append(pairs_text[i][1])
            dpo_data['rejected'].append(pairs_text[i][0])

    return Dataset.from_dict(dpo_data)
