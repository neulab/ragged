import json
import os 
import warnings
import numpy as np
import pdb
import argparse
import matplotlib.pyplot as plt
from file_utils import save_json, save_jsonl, load_jsonl, load_json

def count_jsonl(filename):
    with open(filename, 'r') as f:
        count = sum(1 for _ in f)
    return count

def get_precision(guess_wiki_id_set, gold_wiki_id_set):
    precision = np.mean([[s in gold_wiki_id_set] for s in guess_wiki_id_set])
    return precision

def get_recall(guess_wiki_id_set, gold_wiki_id_set):
    recall = np.mean([[s in guess_wiki_id_set] for s in gold_wiki_id_set]) if len(gold_wiki_id_set) > 0 else 0.0
    return recall

def get_retriever_results(guess_data, gold_data, is_bioasq = False):

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
        
    # guess_data = load_jsonl(guess_file, sort_by_id = True)
    retriever_results = []

    if (len(guess_data) != len(gold_data)):  
        warnings.warn(f"Guess has {len(guess_data)} lines while gold has {len(gold_data)} lines.")
    else: 
            for sample_id, (guess, gold) in enumerate(zip(guess_data, gold_data)):
                guess_sample_id = str(guess['id'])
                gold_sample_id = str(gold['id'])
                gold_wiki_ids = gold['output'][f'{docid_name}_id_set']
                gold_wiki_par_ids = gold['output'][f'{docid_name}_{section_name}_id_set']

                if (guess_sample_id != gold_sample_id):
                    warnings.warn(f"sample {sample_id} return different sample ids {guess_sample_id} from guess and {gold_sample_id} from gold.")
                    break
                
                # For each retrieved document, get wiki and par id match
                doc_retriever_results = []

                for p in guess['output'][0]['provenance']:
                    guess_wiki_id = p[docid_key]
                    
                    doc_retriever_result = {f'{docid_name}_id': guess_wiki_id,
                                        f'{docid_name}_id_match': guess_wiki_id in gold_wiki_ids}
                    
                    guess_wiki_par_id = guess_wiki_id + '_' + str(p[section_key])
                    doc_retriever_result['answer_in_context'] = any([ans in p['text'] for ans in gold['output']['answer_set']])
                    doc_retriever_result[f'{docid_name}_{section_name}_id'] = guess_wiki_par_id
                    doc_retriever_result[f'{docid_name}_{section_name}_id_match'] = guess_wiki_par_id in gold_wiki_par_ids
                    doc_retriever_results.append(doc_retriever_result)

                retriever_result = {'id': guess['id'],\
                                    'gold provenance metadata': {f'num_{docid_name}_ids': len(gold_wiki_ids)},\
                                    'doc-level results': doc_retriever_results}
                retriever_result['gold provenance metadata'][f'num_{docid_name}_{section_name}_ids'] = len(gold_wiki_par_ids)
                retriever_results.append(retriever_result)
    return retriever_results

def print_retriever_acc(retriever_results, gold_data, ks, is_bioasq = False):

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

    
    wiki_id_per_r = []
    wiki_par_id_per_r = []
    answer_in_context_per_r = []

    for r in retriever_results:
        wiki_ids = []
        wiki_par_ids = []
        answer_in_context = []
        
        for d in r['doc-level results']:
            wiki_ids.append(d[f'{docid_name}_id'])
            wiki_par_ids.append(d[f'{docid_name}_{section_name}_id'])
            answer_in_context.append(d['answer_in_context'])

        wiki_id_per_r.append(wiki_ids)
        wiki_par_id_per_r.append(wiki_par_ids)
        answer_in_context_per_r.append(answer_in_context)

    
    results_by_k = {}
    for k in ks:
        # print(k)
        results_by_k[(int)(k)] = {
            f'top-k {docid_name}_id accuracy': 0,\
            f'top-k {docid_name}_{section_name}_id accuracy': 0,\
            f"precision@k {docid_name}_id": 0,\
            f"precision@k {docid_name}_{section_name}_id": 0,\
            f"recall@k {docid_name}_id": 0,\
            f"recall@k {docid_name}_{section_name}_id": 0,\
            'answer_in_context@k': 0
        }
        for r in range(len(retriever_results)):
            gold_wiki_id_set = gold_data[r]['output'][f'{docid_name}_id_set']
            gold_wiki_par_id_set = gold_data[r]['output'][f'{docid_name}_{section_name}_id_set']
            guess_wiki_id_set = set(wiki_id_per_r[r][:k])
            guess_wiki_par_id_set = set(wiki_par_id_per_r[r][:k])
            results_by_k[(int)(k)][f'top-k {docid_name}_id accuracy'] += any([(w in gold_wiki_id_set) for w in guess_wiki_id_set])
            results_by_k[(int)(k)][f'top-k {docid_name}_{section_name}_id accuracy'] += any([(w in gold_wiki_par_id_set) for w in guess_wiki_par_id_set])
            results_by_k[(int)(k)][f"precision@k {docid_name}_id"] += get_precision(guess_wiki_id_set, gold_wiki_id_set)
            results_by_k[(int)(k)][f"precision@k {docid_name}_{section_name}_id"] += get_precision(guess_wiki_par_id_set, gold_wiki_par_id_set)
            results_by_k[(int)(k)][f"recall@k {docid_name}_id"] += get_recall(guess_wiki_id_set, gold_wiki_id_set)
            results_by_k[(int)(k)][f"recall@k {docid_name}_{section_name}_id"] += get_recall(guess_wiki_par_id_set, gold_wiki_par_id_set)
            results_by_k[(int)(k)]["answer_in_context@k"] += any(answer_in_context_per_r[r][:k])

        for key,val in results_by_k[(int)(k)].items():
            results_by_k[(int)(k)][key] = val/len(retriever_results)
    return results_by_k
            
def results_by_key(ks, results_by_k, is_bioasq = False):
    if is_bioasq:
        docid_key = 'pmid'
        docid_name = 'pm'
        section_key = 'section'
        section_name = 'sec'
    else:
        docid_key = 'wikipedia_id'
        docid_name = 'wiki'
        section_name = 'par'
        section_key = 'start_paragraph_id'
    # ks = results_by_k.keys()
    wiki_id_match = []
    wiki_par_id_match = []
    for k in ks:
        wiki_id_match.append(results_by_k[k][f'top-k {docid_name}_id accuracy'])
        wiki_par_id_match.append(results_by_k[k][f'top-k {docid_name}_{section_name}_id accuracy'])
    return wiki_id_match, wiki_par_id_match

def main(model, dataset):
    result_dir = '/data/tir/projects/tir6/general/afreens/dbqa'
    guess_file = os.path.join(result_dir, 'retriever_results/predictions', model, dataset + '.jsonl')
    evaluation_dir = os.path.join(result_dir, 'retriever_results/evaluations', model)
    is_bioasq = True if 'bioasq' in dataset else False

    guess_data = load_jsonl(guess_file, sort_by_id = True)
    if 'bioasq' in dataset:
        gold_data = load_json(os.path.join(result_dir, 'data/gold_zeno_files', f"gold_bioasq_zeno_file.json"), sort_by_id = True)
    else:
        gold_data = load_json(os.path.join(result_dir, 'data/gold_zeno_files', f"gold_{dataset.split('-')[0]}_zeno_file.json"), sort_by_id = True)

    par_retriever_results = get_retriever_results(guess_data, gold_data, is_bioasq = is_bioasq)

    os.makedirs(evaluation_dir, exist_ok=True)
    save_jsonl(par_retriever_results, os.path.join(evaluation_dir, dataset + '.jsonl'))

    # if (args.retriever != 'gold'):
    ks = np.arange(1,101)
    # ks = [1,2,3, 10, 20, 50, 100]
    # ks = [50]
    results_by_k =  print_retriever_acc(par_retriever_results, gold_data, ks, is_bioasq = is_bioasq)
    save_json(results_by_k, os.path.join(evaluation_dir, dataset+ '_results_by_k.json'))

    wiki_id_match, wiki_par_id_match = results_by_key(ks, results_by_k, is_bioasq = is_bioasq)
    plt.title(f'retriever = {model}, dataset = {dataset}')
    plt.ylabel('top-k accuracy')
    plt.plot(ks, wiki_id_match,'b--', label = 'wiki_id')
    plt.plot(ks, wiki_par_id_match, 'b', label = 'wiki_par_id')
    plt.xlabel('k')
    plt.legend()
    plt.savefig(os.path.join(evaluation_dir, dataset+ '_results_by_k.jpg'))
    plt.show()
    print('saving figure in', os.path.join(evaluation_dir, dataset+ '_results_by_k.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--retriever", help="retriever model name")
    parser.add_argument("--dataset", help='datasetname')
    args = parser.parse_args()
    main(args.retriever, args.dataset)