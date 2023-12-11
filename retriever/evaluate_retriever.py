import json
import os 
import warnings
import numpy as np
import pdb
import argparse
import matplotlib.pyplot as plt
from file_utils import write_json

def count_jsonl(filename):
    with open(filename, 'r') as f:
        count = sum(1 for _ in f)
    return count


def load_jsonl(file_path, sort_by_id = True):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            # break
            data.append(json_obj)
    # print('list has', len(data), 'lines')
    if sort_by_id:
        return sorted(data, key=lambda x: x['id'])
    return data


def get_retriever_results(guess_file, gold_file):
    guessfile_len= count_jsonl(guess_file)
    goldfile_len = count_jsonl(gold_file)
    retriever_results = []


    if (guessfile_len != goldfile_len):  
        warnings.warn(f"{guess_file} has {guessfile_len} lines while {gold_file} has {goldfile_len} lines.")
    else: 
        with open(guess_file, 'r') as guessfile, open(gold_file, 'r') as goldfile:
            print(f"Both {guess_file} and {gold_file} have the same number of lines: {guessfile_len}.")
            for sample_id, (guess, gold) in enumerate(zip(guessfile, goldfile)):
                
                guess = json.loads(guess)
                # print(guess)
                gold = json.loads(gold)
                guess_sample_id = str(guess['id'])
                gold_sample_id = str(gold['id'])
                # print(guess_sample_id, gold_sample_id)
                # print(guess_sample_id-gold_sample_id)
                # print(type(guess_sample_id))
                if (guess_sample_id != gold_sample_id):
                    warnings.warn(f"sample {sample_id} return different sample ids {guess_sample_id} from guess and {gold_sample_id} from gold.")
                    break
                
                gold_wiki_ids = set()
                gold_wiki_par_ids = set()

                for ev in gold['output']:
                    if 'provenance' in ev:
                        for prov in ev['provenance']:
                            gold_wiki_id = prov['wikipedia_id'] if 'wikipedia_id' in prov else None
                            if (gold_wiki_id):
                                gold_wiki_ids.add(gold_wiki_id)
                                start_par_id = prov['start_paragraph_id'] if 'start_paragraph_id' in prov else None
                                end_par_id = prov['end_paragraph_id'] if 'end_paragraph_id' in prov else None
                                if (start_par_id and end_par_id):
                                    # pdb.set_trace()
                                    for p_id in range(start_par_id, end_par_id+1):
                                        gold_wiki_par_ids.add(gold_wiki_id + '_' + str(p_id))
                
                # For each retrieved document, get wiki and par id match
                doc_retriever_results = []
                for p in guess['output'][0]['provenance']:
                    guess_wiki_id = p['wikipedia_id']
                    
                    doc_retriever_result = {'wiki_id': guess_wiki_id,
                                        'wiki_id_match': guess_wiki_id in gold_wiki_ids}
                    guess_wiki_par_id = guess_wiki_id + '_' + str(p['start_paragraph_id'])
                    doc_retriever_result['wiki_par_id'] = guess_wiki_par_id
                    doc_retriever_result['wiki_par_id_match'] = guess_wiki_par_id in gold_wiki_par_ids


                    doc_retriever_results.append(doc_retriever_result)
                retriever_result = {'id': guess['id'],
                                        'gold provenance metadata': {'num_wiki_ids': len(gold_wiki_ids)},
                                            'doc-level results': doc_retriever_results}
                retriever_result['gold provenance metadata']['num_wiki_par_ids'] = len(gold_wiki_par_ids)
                retriever_results.append(retriever_result)
    return retriever_results

def print_retriever_acc(retriever_results, ks, par = False):
    
    query_level_wiki_id_match = {}
    query_level_wiki_par_id_match = {}
    doc_level_wiki_id_match = {}
    doc_level_wiki_par_id_match = {}

    for k in ks:
        query_level_wiki_id_match[k] = []
        query_level_wiki_par_id_match[k] = []
        doc_level_wiki_id_match[k] = []
        doc_level_wiki_par_id_match[k] = []

    for s in retriever_results:
        d_wiki_match = []
        d_wiki_par_match = []

        for d in s['doc-level results']:
            d_wiki_match.append(d['wiki_id_match'])
            if par:
                d_wiki_par_match.append(d['wiki_par_id_match'])
        for k in ks:
            query_level_wiki_id_match[k].append(any(d_wiki_match[:k]))
        
            doc_level_wiki_id_match[k].extend(d_wiki_match[:k])
            if par:
                doc_level_wiki_par_id_match[k].extend(d_wiki_par_match[:k])
                query_level_wiki_par_id_match[k].append(any(d_wiki_par_match[:k]))
        
    results_by_k = {}
    for k in ks:
        results_by_k[(int)(k)] = {'top-k wiki_id match accuracy': np.mean(query_level_wiki_id_match[k]), 
                                'top-k wiki_par_id match accuracy': np.mean(query_level_wiki_par_id_match[k]),\
                                'wiki_id match accuracy': np.mean(doc_level_wiki_id_match[k]), \
                                'wiki_par_id match accuracy': np.mean(doc_level_wiki_par_id_match[k])}
    
        # break
    # results_dict = {'query_level_wiki_id_match': query_level_wiki_id_match,
    #                 'query_level_wiki_par_id_match': query_level_wiki_par_id_match, 
    #                 'doc_level_wiki_id_match': doc_level_wiki_id_match, 
    #                 'doc_level_wiki_par_id_match': doc_level_wiki_par_id_match}
    return results_by_k
    # return query_level_wiki_id_match, query_level_wiki_par_id_match, doc_level_wiki_id_match, doc_level_wiki_par_id_match
    

def reformat_guess_file(guess_file, reformat_file):
    with open(guess_file, 'r') as infile, open(reformat_file, 'w') as outfile:
        for l_id, line in enumerate(infile):
            # if (l_id%100000==0):
            #     print(l_id)
            data = json.loads(line)
            # print(data)
            data['output'] = [data['output'][0]]
            # new_prov_set  = []
            # break
            for p in data['output'][0]['provenance']:
                wikipedia_id, paragraph_id = p['docid'].split('_')
                p['wikipedia_id'] = wikipedia_id
                p['start_paragraph_id'] = paragraph_id
                p['end_paragraph_id'] = paragraph_id
                del p['docid']
            # break
            outfile.write(json.dumps(data) + "\n")
            
def results_by_key(ks, results_by_k):
    # ks = results_by_k.keys()
    wiki_id_match = []
    wiki_par_id_match = []
    for k in ks:
        wiki_id_match.append(results_by_k[k]['top-k wiki_id match accuracy'])
        wiki_par_id_match.append(results_by_k[k]['top-k wiki_par_id match accuracy'])
    return wiki_id_match, wiki_par_id_match
    

def main(model, dataset):
    result_dir = "/data/user_data/jhsia2/dbqa"
    guess_file = os.path.join(result_dir, 'retriever_results/predictions', model, dataset + '.jsonl')
    # reformat_file = os.path.join(result_dir, 'retriever_results/predictions', model, 'reformatted-' + dataset + '.jsonl')
    reformat_file = guess_file
    gold_file = os.path.join(result_dir, 'data', dataset + '.jsonl')
    evaluation_dir = os.path.join(result_dir, 'retriever_results/evaluations', model)
    par_retriever_results = get_retriever_results(reformat_file, gold_file)

    os.makedirs(evaluation_dir, exist_ok=True)
    write_json(par_retriever_results, os.path.join(evaluation_dir, dataset + '.jsonl'))
    # with open(os.path.join(evaluation_dir, dataset + '.jsonl'), 'w') as file:
    #     for i, r in enumerate(par_retriever_results):
    #         # if (i%100000==0):
    #         #     print(i)
    #         json.dump(r, file)
    #         file.write('\n')

    ks = np.arange(1,101)
    # ks = [1,2,3, 10, 20, 50, 100]
    # ks = [50]
    # query_level_wiki_id_match, query_level_wiki_par_id_match, doc_level_wiki_id_match, doc_level_wiki_par_id_match =  print_retriever_acc(par_retriever_results, ks, par = True)
    
    results_by_k =  print_retriever_acc(par_retriever_results, ks, par = True)
    with open (os.path.join(evaluation_dir, dataset+ '_results_by_k.json'), 'w') as f:
        json.dump(results_by_k, f, indent = 4)

    wiki_id_match, wiki_par_id_match = results_by_key(ks, results_by_k)
    plt.title(f'retriever = {model}, dataset = {dataset}')
    plt.ylabel('top-k accuracy')
    plt.plot(ks, wiki_id_match,'b--', label = 'wiki_id')
    plt.plot(ks, wiki_par_id_match, 'b', label = 'wiki_par_id')
    plt.xlabel('k')
    plt.legend()
    plt.savefig(os.path.join(evaluation_dir, dataset+ '_results_by_k.jpg'))
    plt.show()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--model", help="retriever model name")
    parser.add_argument("--dataset", help='datasetname')
    args = parser.parse_args()
    main(args.model, args.dataset)