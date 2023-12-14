import numpy as np
import os
import json
from file_utils import load_jsonl, save_jsonl

gold = load_jsonl('/data/user_data/jhsia2/dbqa/data/bioasq/gold_medline_corpus.jsonl')
sampled = load_jsonl('/data/user_data/jhsia2/dbqa/data/bioasq/sampled_medline_corpus.jsonl')

print(len(gold), len(sampled))
combined_list = [l for l in gold] + [l for l in sampled]
combined_list = sorted(combined_list, key=lambda x: x['id'])

save_jsonl(combined_list, '/data/user_data/jhsia2/dbqa/data/medline_corpus_jsonl/medline_corpus.jsonl')