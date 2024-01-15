import sys
sys.path.append('../')

import torch
import transformers
from typing import List

from utils.save_results import save


def inference(prompts: List[str],
              version: str,
              gen_model_src: str = "lvwerra/gpt2-imdb",
              reward_model_src: str = "lvwerra/distilbert-imdb") -> None:
    """
    Inference of the model.

    Args:
        prompts: list of initial prompts
        version: what type of experiment are we handling
        gen_model_src: path to the generative model
        reward_model_src: path to the reward model
    """
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    gen_tokenizer = transformers.AutoTokenizer.from_pretrained(gen_model_src, device_map=device)
    gen_model = transformers.AutoModelForCausalLM.from_pretrained(gen_model_src, device_map=device)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(reward_model_src)
    reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(reward_model_src,
                                                                                   device_map=device)

    input = gen_tokenizer(prompts,
                          return_tensors="pt",
                          padding=True).to(device)
    output = gen_model.generate(**input,
                                max_new_tokens=100,
                                no_repeat_ngram_size=2,
                                do_sample=True,
                                temperature=0.25,
                                num_return_sequences=2)
    generated_texts = []

    for i in range(output.shape[0]):
        generated_text = gen_tokenizer.decode(output[i],
                                              skip_special_tokens=True)
        generated_texts.append(generated_text)

    input_rewards = reward_tokenizer(generated_texts,
                                     return_tensors="pt",
                                     padding=True).to(device)
    with torch.no_grad():
        logits = reward_model(**input_rewards).logits

    save(version, generated_texts, logits, gen_tokenizer)
