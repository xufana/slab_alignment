import os
import sys
import fire
sys.path.append('../')


from src.train import train
from src.inference import inference
from src.make_dataset import prepate_prompts


def main(mode: str,
         n: int,
         version: str,
         template_src: str = 'data/templates.json',
         titles_src: str = 'data/titles.txt',
         prompts_src: str | None = 'data/prompts.txt',
         gen_model: str = "lvwerra/gpt2-imdb",
         reward_model: str = "lvwerra/distilbert-imdb",
         loss_type: str = 'hinge') -> None:
    """
    Args:
        mode: either 'train' or 'inference'
        n: number of generated prompts
        version: whether aligned or not, or other specifications
        description: what type of experiment are we handling
        template_src: path to the file with prompt templates (no needed if prompts_src is not None)
        titles_src: path to the file with movie titles (no needed if prompts_src is not None)
        prompts_src: path to the file with prompts
        gen_model: path to the generative model
        reward_model: path to the reward model
        loss_type: either 'hinge' or 'sigmoid'
    """

    assert loss_type in ['hinge', 'sigmoid']

    # Generating data if there isn't any, reading it otherwise

    if not prompts_src:
        prompts = prepate_prompts(n, template_src, titles_src)
        
        # check-point for prompts
        with open('data/prompts.txt', 'w') as f:
            f.write('\n'.join(prompts))

        prompts_src = 'data/prompts.txt'
    else:
        with open(prompts_src, 'r') as f:
            prompts = f.read().split("\n")

    if mode == 'train':
        train(prompts, loss_type, gen_model, reward_model)
    
    elif mode == 'inference':
        inference(prompts, version, gen_model, reward_model)

    else:
        raise Exception('Wrong mode, use either train or inference!')

    if not os.path.exists('./results'):
        os.makedirs('./results')


if __name__ == "__main__":
    fire.Fire(main)
