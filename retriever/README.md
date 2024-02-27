We support retrievers BM25 and ColBERT and retrieval corpuses kilt_wikipedia (KILT version) and the 2023 Annual Medline Corpus

There are 4 steps: 
1. Download and process corpus data
2. Download query dataset
3. Get retriever outputs
4. Evaluate retriever outputs

# 1. Download and process corpus data

## Download BioAS corpus - Pubmed
<!-- 
        python download_pubmed_corpus.py --data_dir /data/tir/projects/tir6/general/afreens/dbqa/data
        This downloads the pubmed corpus in unprocessed form to ${data_dir}/bioasq/annual_zips/

        use python create_pubmed_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
        This outputs 'pubmed/pubmed_jsonl/pubmed.jsonl' and 'pubmed/id2title.json' in corpus_dir 
        python create_wiki_paragraph_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
    This outputs 'kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl" in your corpus_dir -->

Download pubmed.jsonl from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged) to your ${corpus_dir}/pubmed/pubmed_jsonl/pubmed.jsonl

## Download Wiki corpus
Download wikipedia.jsonl from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged) to your ${corpus_dir}/kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl
    

To process corpus for ColBERT format , run `create_corpus_tsv.py --corpus_name $corpus --corpus_dir $corpus_dir`.
This outputs '${corpus_name}/${corpus_name}.tsv' in corpus_dir

# 2. Download query dataset
Download ${dataset}.jsonl from [here](https://huggingface.co/datasets/jenhsia/ragged) to your ${data_dir}/${dataset}.jsonl
    <!-- Download NQ, hotpotqa from KILT repo as nq.jsonl and hotpotqa.jsonl in the ${data_dir} Download BioASQ
        From Bioasq website, download the following into data_dir/bioasq/
        Task11BGoldenEnriched/11B*_golden.json and BioASQ-training11b/training11b.json from BioASQ
        python compile_bioasq_questions.py --data_dir --corpus_dir 
        This outputs bioasq.jsonl in the data_dir -->
    

To process query tsv for ColBERT, run `python create_query_tsv.py --data_dir $data_dir --dataset $dataset`.
This outputs a {dataset}-queries.tsv

# 3. Get retriever outputs
## Use BM25 for predictions
Download the [pyserini repo](https://github.com/castorini/pyserini) and [KILT repo](https://github.com/facebookresearch/KILT/tree/main/kilt). See more details about bm25 input formatting [here] (https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation).

Customize BM25/default_bm25.json for your select dataset and move the file into KILT/kilt/configs/retriever/default_bm25.json.

Copy BM25/BM25_connector.py into KILT/kilt/retrievers/BM25_connector.py.

Modify KILT/kilt/configs/${dataset}.json for your select dataset.

Run `bm25.sh`

## Use ColBERT for predictions
Download our [modified version](https://github.com/jenhsia/RAGGED_ColBERT) of the [original ColBERT](https://github.com/stanford-futuredata/ColBERT).

Run `colbert.sh`

# 4. Evaluate retriever outputs
To evaluate the predictions, run evaluate_retriver.sh with the appropriate arguments.








