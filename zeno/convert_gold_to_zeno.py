


import csv
import json
import argparse
import os
from file_utils import load_json, save_json, load_jsonl
import pdb


# def convert_gold_to_zeno(input_file, output_file, is_bioasq = False):
def convert_gold_to_zeno(input_file, is_bioasq = False):
    if is_bioasq:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_key = 'start_paragraph_id'
        section_name = 'par'

    # print('total num documents = 5903530')
    num_par_docs = 0
    gold_data_list = []
    # with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
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
                                f'{docid_name}_id_set' : set(), \
                                    f'{docid_name}_{section_name}_id_set': set()}
            
            for output in data['output']:
                ans = output.get('answer', None)
                if (ans):
                    new_data['output']['answer_set'].add(ans)
                provs = output.get('provenance', None)
                if provs:
                    for prov in provs:
                        wiki_id = prov.get(f'{docid_key}', None) 
                        if is_bioasq:
                            start_par_id = 0 if prov.get(f'{section_key}', None)== 'title' else 1
                            # end_par_id = prov.get('end_paragraph_id', None) 
                            end_par_id = start_par_id
                        else:
                            # pdb.set_trace()
                            start_par_id = (int)(prov.get(f'{section_key}', None))
                            end_par_id = start_par_id
                        end_par_id = (int)(start_par_id)
                        title = prov.get('title', None)
                        if(wiki_id):
                            new_data['output'][f'{docid_name}_id_set'].add(wiki_id)
                        # if (start_par_id and end_par_id):
                            for p_id in range(start_par_id, end_par_id+1):
                                new_data['output'][f'{docid_name}_{section_name}_id_set'].add(wiki_id + '_' + str(p_id))
                        if title:
                            new_data['output']['title_set'].add(title)
            for k in new_data['output']:
                new_data['output'][k] = list(new_data['output'][k])
            # outfile.write(json.dumps(new_data) + "\n")
            gold_data_list.append(new_data)
            # break
    return  gold_data_list
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--dataset", help='dataset')
    args = parser.parse_args()
    data_dir = os.path.join('/data/tir/projects/tir6/general/afreens/dbqa/data')
    is_bioasq = (args.dataset == 'bioasq')
    input_file = os.path.join(data_dir, f"{args.dataset}.jsonl")
    dataset = args.dataset.split('_')[0]
    # output_file = os.path.join(data_dir, f"gold-{dataset}.json")
    zeno_format_data = convert_gold_to_zeno(input_file, is_bioasq)
    dataset = args.dataset.split('-')[0]
    save_json(zeno_format_data, os.path.join(data_dir, f'gold_{dataset}_zeno_file.json'))
