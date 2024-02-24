import json
import numpy as np
import random
from file_utils import load_json, save_jsonl, load_jsonl, save_json
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="input dataset name")
    parser.add_argument("--corpus_dir", help='what is the base directory for all corpus files?')
    parser.add_argument("--corpus", help='what is the name of the corpus you are using? wikipedia or pubmed')
    parser.add_argument("--data_dir", help='where is the folder you stored the dataset/dataset jsonl in?')
    parser.add_argument("--dataset", help='what is the name of the dataset/dataset?')
    args = parser.parse_args()

    gold_dataset = []
    for i in range(1,5):
        gold_dataset += load_json(os.path.join(args.data_dir, f'bioasq/Task11BGoldenEnriched/11B{i}_golden.json'))['questions']
    train_dataset = load_json(os.path.join(args.data_dir, 'bioasq/BioASQ-training11b/training11b.json'))
    train_dataset = train_dataset['questions']
    print(len(gold_dataset), len(train_dataset))

    combined_dataset = gold_dataset + train_dataset
    print(len(combined_dataset))

    id2title = load_json(os.path.join(args.corpus_dir, args.corpus, 'id2title.json'))

    questions_categorized = []

    print(len(combined_dataset))
    for i, q in enumerate(combined_dataset):
        if 'exact_answer' not in q.keys():
            continue
        questions_categorized.append(q['type'])

    save_json(questions_categorized, os.path.join(args.data_dir, f'{args.dataset}_questions_categorized.json'))