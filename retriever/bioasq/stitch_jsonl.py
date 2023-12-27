import numpy as np
import os
import json
from file_utils import load_jsonl, save_jsonl

root_dir = '/data/tir/projects/tir6/general/afreens/dbqa/data'

gold = load_jsonl(os.path.join(root_dir, 'bioasq/gold_medline_corpus.jsonl'))
sampled = load_jsonl(os.path.join(root_dir, 'bioasq/sampled_medline_corpus.jsonl'))

print(len(gold), len(sampled))
combined_list = [l for l in gold] + [l for l in sampled]
combined_list = sorted(combined_list, key=lambda x: x['id'])

save_jsonl(combined_list, os.path.join(root_dir, 'medline_corpus_jsonl/medline_corpus.jsonl'))