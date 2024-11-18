from FlagEmbedding import FlagReranker
import argparse
import sys
sys.path.append('/home/jhsia2/ragged')
from file_utils import load_jsonl, save_jsonl
import os
import time
import pdb
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    # retriever = 'colbert'
    # reranker = 'bge_large'
    # dataset = 'nq-dev-kilt'
    # pdb.set_trace()
    if args.reranker == 'bge_large':
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) 
    elif args.reranker == 'bge_v2_m3':
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    retrieved_path = f'/data/tir/projects/tir6/general/afreens/dbqa/retriever_results/predictions/{args.retriever}/{args.dataset}.jsonl'
    retrieved_passages = load_jsonl(retrieved_path)
    reranker_path = f'/data/tir/projects/tir6/general/afreens/dbqa/retriever_results/predictions/{args.retriever}_{args.reranker}_{args.k}/{args.dataset}.jsonl'
    print('reranker path', reranker_path)
    # pdb.set_trace()
    if args.overwrite.lower() == 'true' and os.path.isfile(reranker_path):
        os.remove(reranker_path)
    
    loaded = load_jsonl(reranker_path) if os.path.isfile(reranker_path) else None
    
    loaded_ids = set([l['id'] for l in loaded]) if loaded else set([])
    # pdb.set_trace()
    

    for i in range(len(retrieved_passages)):
        print(i)
        # pdb.set_trace()
        if retrieved_passages[i]['id'] not in loaded_ids:
            query = retrieved_passages[i]['input']
            retrieved_passages[i]['output'][0]['provenance'] = retrieved_passages[i]['output'][0]['provenance'][:args.k]
            texts = [doc['text'] for doc in retrieved_passages[i]['output'][0]['provenance']]
            
            start_time = time.time()
            # print('start computing score')
            scores = reranker.compute_score([[query, passage] for passage in texts])
            # print('done scoring')
            for j, doc in enumerate(retrieved_passages[i]['output'][0]['provenance']):
                doc['score'] = scores[j]
            end_time = time.time()
            print('Time taken: ', (end_time - start_time)/60)
            retrieved_passages[i]['output'][0]['provenance'] = sorted(retrieved_passages[i]['output'][0]['provenance'], key=lambda x: x['score'], reverse=True)
                
            save_jsonl([retrieved_passages[i]], reranker_path,'a')
        # else:
        #     save_jsonl([])
    # save_jsonl(retrieved_passages, f'/data/tir/projects/tir6/general/afreens/dbqa/retriever_results/predictions/{args.retriever}_{args.reranker}/{args.dataset}.jsonl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input, gold, and output files")
    parser.add_argument("--prediction_dir", help="retriever model name", default = '/data/tir/projects/tir6/general/afreens/dbqa/retriever_results/predictions')
    # parser.add_argument("--index_dir", help="retriever model name", default = '/data/tir/projects/tir6/general/afreens/dbqa/indices')
    parser.add_argument("--reranker", default = 'bge_large')
    parser.add_argument("--retriever")
    parser.add_argument("--dataset")
    parser.add_argument("--overwrite", default = 'false')
    parser.add_argument("--k", type=int, default=100)
    # parser.add_argument("--batch_end", type = int, default = -1)
    args = parser.parse_args()
    main(args)
