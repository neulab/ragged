import numpy as np
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import os
import json
import argparse
import pdb

def main(args):
    exp_name = 'colbert'

    with Run().context(RunConfig(nranks=1, experiment=exp_name)):
        config = ColBERTConfig(
            index_path = os.path.join(args.index_dir, 'colbert', args.corpus),
            nbits=2,
            root= args.prediction_dir,
        )
        print('STARTING INDEXING')
        indexer = Indexer(checkpoint=os.path.join(args.model_dir, 'colbertv2.0/'), config=config)
        indexer.index(name=args.corpus, collection=os.path.join(args.corpus_dir, args.corpus, f'{args.corpus}.tsv'), overwrite = 'reuse')
        print('DONE INDEXING')

        searcher = Searcher(index=args.corpus, config=config)
        query_file = os.path.join(args.data_dir, args.dataset + '-queries.tsv')
        print('loading query from', query_file)
        queries = Queries(query_file)
        print('START SEARCHING')
        ranking = searcher.search_all(queries, k=100)

        pred_dir = os.path.join(args.prediction_dir, exp_name)
        os.makedirs(pred_dir, exist_ok = True)

        result_file = os.path.join(pred_dir, f'{args.dataset}.jsonl')
        print('writing search results in', result_file)

        with open(result_file, 'w') as outfile:
            for r_id, r in enumerate(ranking.data):
                r_results = {}
                r_results['id'] = r
                r_results['input'] = queries.data[r]
                r_results['output'] = [{'provenance': []}]
                for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                    page_id, paragraph_id = searcher.collection.pid_list[c_id].split('_')
                    r_results['output'][0]['provenance'].append({"page_id": page_id,\
                                                                "par_id": paragraph_id,\
                                                                "text": searcher.collection.data[c_id],\
                                                                "score": score})
                json.dump(r_results, outfile)
                outfile.write('\n')
        print('DONE')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", help='which directory are you saving the colbert predictions in?')
    parser.add_argument("--model_dir", help='where did you save the colbert model you downloaded?')
    parser.add_argument("--corpus_dir", help='what is the base directory for all corpus files?')
    parser.add_argument("--corpus", help='what is the name of the corpus you are using? wikipedia or pubmed')
    parser.add_argument("--data_dir", help='where is the folder you stored the dataset jsonl in?')
    parser.add_argument("--dataset", help='what is the name of the dataset?')
    parser.add_argument("--index_dir", help='what is the name of the colbert index dir?')
    args = parser.parse_args()
    main(args)