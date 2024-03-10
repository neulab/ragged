


import csv
import json
import argparse
import os
from file_utils import load_json, save_json, load_jsonl
import pdb

def create_gold_file(input_file):
    gold_data_list = []

    with open(input_file, 'r') as infile:
        for l_id, line in enumerate(infile):
            if (l_id%1000==0):
                print(l_id)
            data = json.loads(line)
            new_data = {}
            new_data['id'] = data['id']
            # print(data)
            new_data['input'] = data['input']
            new_data['output'] = {'answer_set' : set(), \
                                'title_set': set(), \
                                'page_id_set' : set(), \
                                'page_par_id_set': set()}
            
            for output in data['output']:
                ans = output.get('answer', None)
                if (ans):
                    new_data['output']['answer_set'].add(ans)
                
                provs = output.get('provenance', None)
                if provs:
                    for prov in provs:
                        page_id = prov.get('page_id', None) 
                        start_par_id = (int)(prov.get('start_par_id', None))
                        end_par_id = (int)(start_par_id)
                        title = prov.get('title', None)
                        if(page_id):
                            new_data['output']['page_id_set'].add(page_id)
                            for par_id in range(start_par_id, end_par_id+1):
                                new_data['output']['page_par_id_set'].add(page_id + '_' + str(par_id))
                        if title:
                            new_data['output']['title_set'].add(title)
            for k in new_data['output']:
                new_data['output'][k] = list(new_data['output'][k])
            gold_data_list.append(new_data)
    return  gold_data_list
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help='where is the folder you stored the dataset jsonl in?')
    parser.add_argument("--dataset", help='what is the name of the dataset?')
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.dataset}.jsonl")
    gold_data = create_gold_file(input_file)
    dataset = args.dataset.split('-')[0]
    os.makedirs(os.path.join(args.data_dir, 'gold_compilation_files'), exist_ok=True)
    save_json(gold_data, os.path.join(args.data_dir, 'gold_compilation_files', f'gold_{args.dataset}_compilation_file.json'))
