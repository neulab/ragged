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

def get_precision(guess_page_id_set, gold_page_id_set):
    precision = np.mean([[s in gold_page_id_set] for s in guess_page_id_set])
    return precision

def get_recall(guess_page_id_set, gold_page_id_set):
    recall = np.mean([[s in guess_page_id_set] for s in gold_page_id_set]) if len(gold_page_id_set) > 0 else 0.0
    return recall

def get_retriever_results(guess_data, gold_data):
        
    retriever_results = []

    if (len(guess_data) != len(gold_data)):  
        warnings.warn(f"Guess has {len(guess_data)} lines while gold has {len(gold_data)} lines.")
    else: 
            for sample_id, (guess, gold) in enumerate(zip(guess_data, gold_data)):
                guess_sample_id = str(guess['id'])
                gold_sample_id = str(gold['id'])
                gold_page_ids = gold['output'][f'page_id_set']
                gold_page_par_ids = gold['output'][f'page_par_id_set']

                if (guess_sample_id != gold_sample_id):
                    warnings.warn(f"sample {sample_id} return different sample ids {guess_sample_id} from guess and {gold_sample_id} from gold.")
                    break
                
                # For each retrieved document, get wiki and par id match
                doc_retriever_results = []

                for p in guess['output'][0]['provenance']:
                    guess_page_id = p['page_id']
                    
                    doc_retriever_result = {f'page_id': guess_page_id,
                                        f'page_id_match': guess_page_id in gold_page_ids}
                    
                    guess_page_par_id = guess_page_id + '_' + str(p['start_par_id'])
                    doc_retriever_result['answer_in_context'] = any([ans in p['text'] for ans in gold['output']['answer_set']])
                    doc_retriever_result[f'page_par_id'] = guess_page_par_id
                    doc_retriever_result[f'page_par_id_match'] = guess_page_par_id in gold_page_par_ids
                    doc_retriever_results.append(doc_retriever_result)

                retriever_result = {'id': guess['id'],\
                                    'gold provenance metadata': {f'num_page_ids': len(gold_page_ids)},\
                                    'paragraph-level results': doc_retriever_results}
                retriever_result['gold provenance metadata'][f'num_page_par_ids'] = len(gold_page_par_ids)
                retriever_results.append(retriever_result)
    return retriever_results

def print_retriever_acc(retriever_results, gold_data, ks):

    page_id_per_r = []
    page_par_id_per_r = []
    answer_in_context_per_r = []

    for r in retriever_results:
        page_ids = []
        page_par_ids = []
        answer_in_context = []
        
        for d in r['paragraph-level results']:
            page_ids.append(d[f'page_id'])
            page_par_ids.append(d[f'page_par_id'])
            answer_in_context.append(d['answer_in_context'])

        page_id_per_r.append(page_ids)
        page_par_id_per_r.append(page_par_ids)
        answer_in_context_per_r.append(answer_in_context)

    
    results_by_k = {}
    for k in ks:
        results_by_k[(int)(k)] = {
            f'top-k page_id accuracy': 0,\
            f'top-k page_par_id accuracy': 0,\
            f"precision@k page_id": 0,\
            f"precision@k page_par_id": 0,\
            f"recall@k page_id": 0,\
            f"recall@k page_par_id": 0,\
            'answer_in_context@k': 0
        }
        for r in range(len(retriever_results)):
            gold_page_id_set = gold_data[r]['output'][f'page_id_set']
            gold_page_par_id_set = gold_data[r]['output'][f'page_par_id_set']
            guess_page_id_set = set(page_id_per_r[r][:k])
            guess_page_par_id_set = set(page_par_id_per_r[r][:k])
            results_by_k[(int)(k)][f'top-k page_id accuracy'] += any([(w in gold_page_id_set) for w in guess_page_id_set])
            results_by_k[(int)(k)][f'top-k page_par_id accuracy'] += any([(w in gold_page_par_id_set) for w in guess_page_par_id_set])
            results_by_k[(int)(k)][f"precision@k page_id"] += get_precision(guess_page_id_set, gold_page_id_set)
            results_by_k[(int)(k)][f"precision@k page_par_id"] += get_precision(guess_page_par_id_set, gold_page_par_id_set)
            results_by_k[(int)(k)][f"recall@k page_id"] += get_recall(guess_page_id_set, gold_page_id_set)
            results_by_k[(int)(k)][f"recall@k page_par_id"] += get_recall(guess_page_par_id_set, gold_page_par_id_set)
            results_by_k[(int)(k)]["answer_in_context@k"] += any(answer_in_context_per_r[r][:k])

        for key,val in results_by_k[(int)(k)].items():
            results_by_k[(int)(k)][key] = val/len(retriever_results)
    return results_by_k
            
def results_by_key(ks, results_by_k):
    page_id_match = []
    page_par_id_match = []
    for k in ks:
        page_id_match.append(results_by_k[k][f'top-k page_id accuracy'])
        page_par_id_match.append(results_by_k[k][f'top-k page_par_id accuracy'])
    return page_id_match, page_par_id_match

def main(args):
    guess_file = os.path.join(args.prediction_dir, args.retriever, args.dataset + '.jsonl')
    evaluation_dir = os.path.join(args.evaluation_dir, args.retriever)

    guess_data = load_jsonl(guess_file, sort_by_id = True)
    gold_data = load_json(os.path.join(args.data_dir, 'gold_zeno_files', f"gold_{args.dataset}_zeno_file.json"), sort_by_id = True)

    par_retriever_results = get_retriever_results(guess_data, gold_data)

    os.makedirs(evaluation_dir, exist_ok=True)
    save_jsonl(par_retriever_results, os.path.join(evaluation_dir, args.dataset + '.jsonl'))

    ks = np.arange(1,101)
    results_by_k =  print_retriever_acc(par_retriever_results, gold_data, ks)
    save_json(results_by_k, os.path.join(evaluation_dir, f'{args.dataset}_results_by_k.json'))

    page_id_match, page_par_id_match = results_by_key(ks, results_by_k)
    plt.title(f'retriever = {args.retriever}, dataset = {args.dataset}')
    plt.ylabel('top-k accuracy')
    plt.plot(ks, page_id_match,'b--', label = 'page_id')
    plt.plot(ks, page_par_id_match, 'b', label = 'page_par_id')
    plt.xlabel('k')
    plt.legend()
    plt.savefig(os.path.join(evaluation_dir, f'{args.dataset}_results_by_k.jpg'))
    plt.show()
    print('saving figure in', os.path.join(evaluation_dir, f'{args.dataset}_results_by_k.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--prediction_dir", help="retriever model name")
    parser.add_argument("--evaluation_dir", help="retriever model name")
    parser.add_argument("--retriever", help="retriever model name")
    parser.add_argument("--dataset", help='datasetname')
    parser.add_argument("--data_dir", help='datasetname')
    args = parser.parse_args()
    main(args)