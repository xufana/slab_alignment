import torch
import transformers
from typing import List
from trl import DPOTrainer


from make_dataset import prepare_dataset


def train(prompts: List[str],
          gen_model_src: str = "lvwerra/gpt2-imdb",
          reward_model_src: str = "lvwerra/distilbert-imdb",
          loss_type: str = 'hinge') -> None:
    """
    Training the model with DPO Trainer.

    Args:
        prompts: list of initial prompts;
        gen_model_src: path to the file with prompt templates
        reward_model_src: path to the file with movie titles
        loss_type: either 'hinge' or 'sigmoid'
    """

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("Initializing models...")

    gen_tokenizer = transformers.AutoTokenizer.from_pretrained(gen_model_src, device_map=device)
    gen_model = transformers.AutoModelForCausalLM.from_pretrained(gen_model_src, device_map=device)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(reward_model_src)
    reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(reward_model_src,
                                                                                   device_map=device)
    
    print("Generating reviews...")
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

    print("Scoring generations...")
    for i in range(output.shape[0]):
        generated_text = gen_tokenizer.decode(output[i],
                                              skip_special_tokens=True)
        generated_texts.append(generated_text)

    input_rewards = reward_tokenizer(generated_texts,
                                     return_tensors="pt",
                                     padding=True).to(device)
    with torch.no_grad():
        logits = reward_model(**input_rewards).logits

    training_args = transformers.TrainingArguments(
        output_dir="../models",
        remove_unused_columns=False)

    dataset = prepare_dataset(prompts, generated_texts, logits)

    print("Training the model...")
    dpo_trainer = DPOTrainer(
        gen_model,
        reward_model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=gen_tokenizer,
        loss_type=loss_type,
    )

    dpo_trainer.train()

    print("Saving the model...")
    dpo_trainer.save_model(f"../models/model_{loss_type}")

    print("All done!")
