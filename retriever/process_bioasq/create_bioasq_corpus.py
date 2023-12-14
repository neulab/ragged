import json
import numpy as np
import random

from file_utils import save_json, load_json
import requests
from bs4 import BeautifulSoup
import os
import numpy as np

medline_corpus_size = 36_475_584

random.seed(42)
print('shuffling corpus of size', medline_corpus_size)
# medline_corpus = np.arange(1, medline_corpus_size+1)
# random.shuffle(medline_corpus)

medline_corpus = np.load('medline_indices.npy')

gold_dataset = []
for i in range(1,5):
    # gold_dataset = 
    gold_dataset += load_json(f'/data/user_data/jhsia2/dbqa/data/bioasq/Task11BGoldenEnriched/11B{i}_golden.json')['questions']
train_dataset = load_json('/data/user_data/jhsia2/dbqa/data/bioasq/BioASQ-training11b/training11b.json')
train_dataset = train_dataset['questions']
print(len(gold_dataset), len(train_dataset))

combined_dataset = gold_dataset + train_dataset
print(len(combined_dataset))

prov_docs = []
doc_ids = set()
for q in combined_dataset:
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

    
    for s in q['snippets']:
        
        docid = s['document'].split('pubmed/')[1]
        beginSec = s['beginSection']
        
        sample_dict['output'].append({'provenance': [{'pmid': docid, \
                        'section': beginSec}]})
        # sample_dict['output'].append({'provenance': [{'docid': docid, \
        #                 'section': beginSec}]})
        prov_docs.append(sample_dict)
        doc_ids.add((int)(docid))

def get_docs(pmid, strict = True):
    url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = None
    abstract = None
    if soup.find('h1', class_='heading-title'):
        title = soup.find('h1', class_='heading-title').get_text().strip()
    if soup.find('div', class_='abstract-content selected'):
        abstract = soup.find('div', class_='abstract-content selected').get_text().strip()
    if title == '':
        title = None
    if abstract == '':
        abstract = None
    
    if strict:
        if title == None or abstract == None:
            return None, None
    return title, abstract

# os.makedirs('/data/user_data/jhsia2/dbqa/data/medline_corpus', exist_ok= True)

# num_added = 0 
missing_title = []
missing_abstract = []
with open('/data/user_data/jhsia2/dbqa/data/bioasq/gold_medline_corpus.jsonl', 'w') as outfile:
    print('adding prov docs')
    for i, pmid in enumerate(doc_ids):
        if (i%100 == 0):
            print(f'{i}/{len(doc_ids)}')
        title, abstract = get_docs(pmid, strict = False)
        if title:
            outfile.write(json.dumps({"id": f'{pmid}_0', "contents": title }) + "\n")
        else:
            missing_title.append(pmid)
        if abstract:
            outfile.write(json.dumps({"id": f'{pmid}_1', "contents": abstract }) + "\n")
        else:
            missing_abstract.append(pmid)

save_json({'missing title': missing_title, 'missing abstract': missing_abstract}, '/data/user_data/jhsia2/dbqa/data/bioasq/gold_corrupt.json')

num_added = 0 
with open('/data/user_data/jhsia2/dbqa/data/medline_corpus/sampled_medline_corpus.jsonl', 'w') as outfile:
    print('adding sampled docs')
    for i, pmid in enumerate(medline_corpus):
        if pmid in doc_ids:
            continue
        title, abstract = get_docs(pmid, strict = True)
        if (title):
            num_added += 1
            print(f'{num_added}/100_000')
            outfile.write(json.dumps({"id": f'{pmid}_0', "contents": title }) + "\n")
            outfile.write(json.dumps({"id": f'{pmid}_1', "contents": abstract }) + "\n")
        if (num_added == 100_000):
            break
