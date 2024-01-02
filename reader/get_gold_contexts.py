


import csv
import json
import argparse
import os
from file_utils import load_json, save_json, load_jsonl, save_jsonl
import pdb


def convert_gold_to_zeno(gold_info, corpus_file, is_bioasq):
    # wiki_par_ids_data = load_json("/data/tir/projects/tir6/general/afreens/dbqa/data/gold-nq-dev-kilt.json")
    # gold_data = load_jsonl(gold_file)
    # tsv_file = "/data/tir/projects/tir6/general/afreens/dbqa/data/kilt_knowledgesource.tsv"

    if is_bioasq:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_key = 'paragraph_id'
        section_name = 'par'

    par_id_to_text_map = {}

    # with open("/data/tir/projects/tir6/general/afreens/dbqa/data/kilt_knowledgesource.tsv") as fd:
    #     rd = csv.reader(fd, delimiter="\t", quotechar='"')
    #     for i, row in enumerate(rd):
    #         if (i%100_000 == 0 ):
    #             print(i)
    #         par_id_to_text_map[row[0]] = row[1]
    #         # break
    # with open("/data/tir/projects/tir6/general/afreens/dbqa/data/kilt_knowledgesource.tsv", 'r') as file:
    # if 'wiki' in corpus_file:
    #     encoding = 'UTF-8'
    # else:
    #     encoding = 'unicode_escape'
    print('loading from corpus file', corpus_file)
    ids = []
    with open(corpus_file, 'r', encoding = 'UTF-8') as file:
        for i, line in enumerate(file):
            if (i%100_000 == 0 ):
                print(i, flush = True)     
            line = line.strip()
            if is_bioasq:
                line = line.encode().decode('unicode_escape')
            split_out = line.split('\t')
            id = split_out[0]
            # pdb.set_trace()
            text = '\t'.join(split_out[1:])
            par_id_to_text_map[id] = text
            ids.append(id)
    # print(len(ids))
    print(len(par_id_to_text_map))

    # zeno_format_data = []
    # output_dicts = []
    
    retriever_format_data = []

    for i, ques_info in enumerate(gold_info):
        if (i%1000 == 0):
            print(i)
        qid = ques_info["id"]
        question = ques_info["input"]
        doc_par_id_set = ques_info["output"][f'{docid_name}_{section_name}_id_set']
        provenance_set = []
        for doc_par_id in doc_par_id_set:
            doc_id, par_id = doc_par_id.split('_')
            output_dict = {}
            output_dict[docid_key] = doc_id
            if is_bioasq:
                output_dict[section_key] = par_id
            else:
                output_dict[f'start_{section_key}'] = par_id
                output_dict[f'end_{section_key}'] = par_id
            output_dict['text'] = par_id_to_text_map.get(doc_par_id, None)
            output_dict['score'] = None
            # if "answer" not in answer or "provenance" not in answer:
            #     continue
            # for p in answer["provenance"]:
                
                # if str(p["wikipedia_id"])+"_"+str(p["start_paragraph_id"]+1) not in par_id_to_text_map.keys():
                #     print(qid)
                    # pdb.set_trace()
                # p["text"] = par_id_to_text_map[str(p["wikipedia_id"])+"_"+str(p["start_paragraph_id"]+1)]

            if (output_dict['text']):
                provenance_set.append(output_dict)
            
        retriever_format_data.append({
            "id": qid,
            "input": question,
            "output": [{'provenance': provenance_set}]
        })
        # break
    return retriever_format_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='nq-dev-kilt or hotpotqa-')
    # parser.add_argument("--corpus", help = 'wiki_par OR medline_tsv')
    args = parser.parse_args()
    root_dir = '/data/tir/projects/tir6/general/afreens/dbqa'
    data_dir = os.path.join(root_dir, 'data')
    dataset = args.dataset.split('-')[0]
    gold_info = load_json(os.path.join(data_dir, 'gold_zeno_files', f'gold_{dataset}_zeno_file.json'))
    
    is_bioasq = 'bioasq' in args.dataset
    if is_bioasq:
        corpus_file  = os.path.join(data_dir, 'corpus_files', f'complete_medline_corpus.tsv')
    else:
        corpus_file  = os.path.join(data_dir, 'corpus_files', f'wiki_par.tsv')
    retriever_format_data = convert_gold_to_zeno(gold_info, corpus_file, is_bioasq)
    # dataset = args.dataset.split('_')[0]

    gold_dir = os.path.join('/data/user_data/jhsia2/dbqa', 'retriever_results/predictions/gold')
    os.makedirs(gold_dir, exist_ok = True)
    save_jsonl(retriever_format_data, os.path.join(gold_dir, f'{args.dataset}.jsonl'))