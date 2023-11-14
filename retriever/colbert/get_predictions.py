import numpy as np
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import os
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import pdb
import json
import argparse

def main(dataset):
    exp_name = 'colbert'
    root_dir = "/data/user_data/jhsia2/dbqa"
    with Run().context(RunConfig(nranks=1, experiment=exp_name)):

        config = ColBERTConfig(
            nbits=2,
            root=os.path.join(root_dir, "retriever_results/predictions"),
        )
        print('STARTING INDEXING')
        indexer = Indexer(checkpoint=os.path.join(root_dir, 'models/colbertv2.0/'), config=config)
        indexer.index(name="msmarco.nbits=2", collection=os.path.join(root_dir, "data/kilt_knowledgesource.tsv"), overwrite = 'reuse')
        print('DONE INDEXING')

        searcher = Searcher(index="msmarco.nbits=2", config=config)
        query_file = os.path.join(root_dir, 'data', dataset + '-queries.tsv')
        print('loading query from', query_file)
        queries = Queries(query_file)
        print('START SEARCHING')
        ranking = searcher.search_all(queries, k=100)

        pred_dir = os.path.join(root_dir, 'retriever_results/predictions', exp_name)
        os.makedirs(pred_dir, exist_ok = True)
        result_file = os.path.join(pred_dir, dataset+ '.jsonl')
        print('writing search results in', result_file)
        with open(result_file, 'w') as outfile:
            for r_id, r in enumerate(ranking.data):
                r_results = {}
                r_results['id'] = r
                r_results['input'] = queries.data[r]
                r_results['output'] = [{'provenance': []}]
                for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                    wikipedia_id, paragraph_id = searcher.collection.pid_list[c_id].split('_')
                    # pdb.set_trace()
                    r_results['output'][0]['provenance'].append({"wikipedia_id": wikipedia_id,\
                            "start_paragraph_id": paragraph_id,\
                                "end_paragraph_id": paragraph_id,\
                                "text": searcher.collection.data[c_id],\
                                "score": score})
                json.dump(r_results, outfile)
                outfile.write('\n')
        print('DONE')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="input dataset name")
    parser.add_argument("dataset", help='dataset name')
    args = parser.parse_args()
    main(args.dataset)