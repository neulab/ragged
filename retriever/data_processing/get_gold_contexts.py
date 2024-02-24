


import csv
import json
import argparse
import os
from file_utils import load_json, save_json, load_jsonl, save_jsonl
import pdb

def convert_gold_to_zeno(gold_info, corpus_file):

    par_id_to_text_map = {}

    print('loading from corpus file', corpus_file)
    ids = []
    with open(corpus_file, 'r', encoding = 'UTF-8') as file:
        for i, line in enumerate(file):
            if (i%100_000 == 0 ):
                print(i, flush = True)     
            line = line.strip()
            line = line.encode().decode('unicode_escape')
            split_out = line.split('\t')
            id = split_out[0]
            text = '\t'.join(split_out[1:])
            par_id_to_text_map[id] = text
            ids.append(id)
    # print(len(par_id_to_text_map))

    retriever_format_data = []

    for i, ques_info in enumerate(gold_info):
        if (i%1000 == 0):
            print(i)
        qid = ques_info["id"]
        question = ques_info["input"]
        page_par_id_set = ques_info["output"]['page_par_id_set']
        provenance_set = []
        for page_par_id in page_par_id_set:
            page_id, par_id = page_par_id.split('_')
            output_dict = {'page_id': page_id, \
                           'start_par_id': par_id,\
                            'end_par_id': par_id,\
                            'text': par_id_to_text_map.get(page_par_id, None), \
                            'score': None}
            if (output_dict['text']):
                provenance_set.append(output_dict)
            
        retriever_format_data.append({
            "id": qid,
            "input": question,
            "output": [{'provenance': provenance_set}]
        })
    return retriever_format_data
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", help='which directory are you saving the colbert predictions in?')
    parser.add_argument("--model_dir", help='where did you save the colbert model you downloaded?')
    parser.add_argument("--corpus_dir", help='what is the base directory for all corpus files?')
    parser.add_argument("--corpus", help='what is the name of the corpus you are using? wikipedia or pubmed')
    parser.add_argument("--data_dir", help='where is the folder you stored the dataset jsonl in?')
    parser.add_argument("--dataset", help='what is the name of the dataset?')
    args = parser.parse_args()

    gold_info = load_json(os.path.join(args.data_dir, 'gold_zeno_files', f'gold_{args.dataset}_zeno_file.json'))
    
    corpus_file  = os.path.join(args.corpus_dir, args.corpus, f'{args.corpus}.tsv')
    retriever_format_data = convert_gold_to_zeno(gold_info, corpus_file)

    gold_dir = os.path.join(args.prediction_dir, 'gold')
    os.makedirs(gold_dir, exist_ok = True)
    save_jsonl(retriever_format_data, os.path.join(gold_dir, f'{args.dataset}.jsonl'))