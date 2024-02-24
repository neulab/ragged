import json
import numpy as np
import random
from file_utils import load_json, save_jsonl, load_jsonl
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--data_dir")
    parser.add_argument("--corpus_dir")
    args = parser.parse_args()
    
    gold_dataset = []
    for i in range(1,5):
        gold_dataset += load_json(os.path.join(args.data_dir, f'bioasq/Task11BGoldenEnriched/11B{i}_golden.json'))['questions']
    train_dataset = load_json(os.path.join(args.data_dir, 'bioasq/BioASQ-training11b/training11b.json'))
    train_dataset = train_dataset['questions']
    
    # print(len(gold_dataset), len(train_dataset))

    combined_dataset = gold_dataset + train_dataset
    # print(len(combined_dataset))

    id2title = load_json(os.path.join(args.corpus_dir, 'pubmed/id2title.json'))

    prov_docs = []
    doc_par_ids = set()

    for i, q in enumerate(combined_dataset):
        if 'exact_answer' not in q.keys():
            continue
        qid = q['id']
        sample_dict = {'id': qid, \
                    'input': q['body'],\
                    'output': []
                    }
        
        answer_set = set()
        if type(q['exact_answer']) == list:
            for a_set in q['exact_answer']:
                if (type(a_set) == list):
                    for a in a_set:
                        answer_set.add(a)
                else:
                    answer_set.add(a_set)
        else:
            answer_set.add(q['exact_answer'])
            
        for a in answer_set:
            sample_dict['output'].append({'answer': a})

        num_prov = 0
        for s in q['snippets']:
            
            docid = s['document'].split('pubmed/')[1]
            beginSec = 0 if s['beginSection'] =='title' else 1
            title = id2title.get(docid, None)
            if title:
                num_prov +=1 
                sample_dict['output'].append({'provenance': [{
                                'page_id': docid, \
                                'start_par_id': beginSec,\
                                'end_par_id': beginSec,\
                                'title': title}]})

        prov_docs.append(sample_dict)

    print(len(prov_docs))
    save_jsonl(prov_docs, os.path.join(args.data_dir, 'bioasq.jsonl'))
