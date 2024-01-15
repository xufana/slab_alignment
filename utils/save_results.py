import csv
import os
import pandas as pd
from utils.diversity import token_entropy


def save(version, texts, logits, tokenizer):
    path = './results/'
    # logits for every generated text, supposed to specify if we alligned or not
    df_logits = pd.DataFrame({'texts': texts, 'logits': logits.tolist()})
    df_logits.to_csv(path + f'{version}_logits.csv', index=False)

    diversity = token_entropy(texts, tokenizer)
    if not os.path.exists(path + 'diversity.csv'):
        with open(path + 'diversity.csv', 'w', newline='') as diversity_csv:
            diversity_csv_write = csv.writer(diversity_csv)
            diversity_csv_write.writerow(
                ["version", "diversity"])

    with open(path + 'diversity.csv', 'a', newline='') as order_csv:
        order_csv_append = csv.writer(order_csv)
        order_csv_append.writerow([version, diversity])
