import json
import numpy as np
import random
from file_utils import load_json, save_jsonl, load_jsonl
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

# np.save()
gold_dataset = []
for i in range(1,5):
    # gold_dataset = 
    gold_dataset += load_json(f'/data/user_data/jhsia2/dbqa/data/bioasq/Task11BGoldenEnriched/11B{i}_golden.json')['questions']
train_dataset = load_json('/data/user_data/jhsia2/dbqa/data/bioasq/BioASQ-training11b/training11b.json')
train_dataset = train_dataset['questions']
print(len(gold_dataset), len(train_dataset))

combined_dataset = gold_dataset + train_dataset
print(len(combined_dataset))

corpus = load_jsonl('/data/user_data/jhsia2/dbqa/data/bioasq/gold_medline_corpus.jsonl')

id2title = {}
for c in corpus:
    docid, sec = c['id'].split('_')
    if (int)(sec) == 0:
        title = c['contents']
        id2title[docid] = title


prov_docs = []
doc_par_ids = set()
print(len(combined_dataset))
for i, q in enumerate(combined_dataset):
    # if (i%1000 == 0):
    #     print('st', i)
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
            for a in a_set:
                answer_set.add(a)
    else:
        answer_set.add(q['exact_answer'])
        
    for a in answer_set:
        sample_dict['output'].append({'answer': a})

    num_prov = 0
    for s in q['snippets']:
        
        docid = s['document'].split('pubmed/')[1]
        beginSec = s['beginSection']
        # title = s['title']
        title = id2title.get(docid, None)
        if title:
            num_prov +=1 
            sample_dict['output'].append({'provenance': [{
                            'pmid': docid, \
                            'section': beginSec,\
                            'title': title}]})
        # sample_dict['output'].append({'provenance': [{'docid': docid, \
        #                 'section': beginSec}]})
    # if (i%1000 == 0):
    #     print('end', i)
    prov_docs.append(sample_dict)
    # if num_prov == 0:
    #     print('qid missing provs', qid)
        # doc_ids.add((int)(docid))
print(len(prov_docs))
save_jsonl(prov_docs, '/data/user_data/jhsia2/dbqa/data/bioasq.jsonl')
