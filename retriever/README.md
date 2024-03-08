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
        python create_page_paragraph_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
    This outputs 'kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl" in your corpus_dir -->

Download pubmed.jsonl from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged) to `${corpus_dir}/pubmed/pubmed_jsonl/pubmed.jsonl`.

## Download Wiki corpus
Download wikipedia.jsonl from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged) to `${corpus_dir}/kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl`.
    

To process corpus for ColBERT format , run `create_corpus_tsv.py --corpus $corpus --corpus_dir $corpus_dir`.
This outputs `$corpus_dir/${corpus}/${corpus}.tsv`.

# 2. Download query dataset
Download each `$dataset` from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged) to `${data_dir}/${dataset}.jsonl`.
    <!-- Download NQ, hotpotqa from KILT repo as nq.jsonl and hotpotqa.jsonl in the ${data_dir} Download BioASQ
        From Bioasq website, download the following into data_dir/bioasq/
        Task11BGoldenEnriched/11B*_golden.json and BioASQ-training11b/training11b.json from BioASQ
        python compile_bioasq_questions.py --data_dir --corpus_dir 
        This outputs bioasq.jsonl in the data_dir -->
    

To process query tsv for ColBERT, run `python create_query_tsv.py --data_dir $data_dir --dataset $dataset`, which outputs `$data_dir/${dataset}-queries.tsv`.

# 3. Get retriever outputs
## Use BM25 for predictions
1. Git clone the [pyserini repo](https://github.com/castorini/pyserini) and [KILT repo](https://github.com/facebookresearch/KILT/tree/main). See more details about BM25 input formatting [here] (https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation).

2. Customize [`BM25/default_bm25.json`](https://github.com/neulab/ragged/blob/main/retriever/BM25/default_bm25.json) for your select dataset and move the file into `KILT/kilt/configs/retriever/default_bm25.json`.

3. Copy [`BM25/BM25_connector.py`](https://github.com/neulab/ragged/blob/main/retriever/BM25/BM25_connector.py) into `KILT/kilt/retrievers/BM25_connector.py`.

4. Customize `KILT/kilt/configs/${dataset}.json` for your select dataset.

5. Run `bm25.sh` to output `${prediction_dir}/bm25/${dataset}.jsonl`.

Each line corresponds to each query. This is an example of one line:
```
{"id": "-1027463814348738734", 
"input": "pace maker is associated with which body organ", 
"output": [{"provenance": [
    # top 1 retrieved paragraph
    {"page_id": "557054", 
    "start_par_id": "4", 
    "end_par_id": "4", 
    "text": "Peristalsis of the smooth muscle originating in pace-maker cells originating in the walls of the calyces propels urine through the renal pelvis and ureters to the bladder. The initiation is caused by the increase in volume that stretches the walls of the calyces. This causes them to fire impulses which stimulate rhythmical contraction and relaxation, called peristalsis. Parasympathetic innervation enhances the peristalsis while sympathetic innervation inhibits it.", 
    "score": 20.9375},
    # top 2 retrieved paragraph ...
    ]}]}
```


## Use ColBERT for predictions
1. Download our [modified version](https://github.com/jenhsia/RAGGED_ColBERT) of the [original ColBERT](https://github.com/stanford-futuredata/ColBERT).

2. Run `colbert.sh` to output `${prediction_dir}/colbert/${dataset}.jsonl`.

# 4. Evaluate retriever outputs
To evaluate the predictions, run \evaluate_retriver.sh' with the appropriate arguments.
This outputs 3 files.
The first output file is `${evaluation_dir}/${retriever}/${dataset}.jsonl`, for which each line corresponds to each query. 
For each query, we include paragraph-level results for each of the k retrieved paragraphs. We include an example of one line below, 
where 'id' corresponds to the query id.

```
{"id": "-1027463814348738734",
"gold provenance metadata": {"num_page_ids": 2, "num_page_par_ids": 2}, 
"paragraph-level results": [
    {"page_id": "557054", "page_id_match": false, "answer_in_context": false, "page_par_id": "557054_4", "page_par_id_match": false},
    ...
    {"page_id": "12887799", "page_id_match": false, "answer_in_context": true, "page_par_id": "12887799_2", "page_par_id_match": false}
    ]
}
```

The second output file is `${evaluation_dir}/${retriever}/${dataset}_results_by_k.json`.
This outputs retrieval performance for each k. We include an example below for k = 1.
```
    "1": {
        "top-k wiki_id accuracy": 0.3595347197744096,
        "top-k wiki_par_id accuracy": 0.24814945364821994,
        "precision@k wiki_id": 0.3595347197744096,
        "precision@k wiki_par_id": 0.24814945364821994,
        "recall@k wiki_id": 0.27195394195746725,
        "recall@k wiki_par_id": 0.12315278912917237,
        "answer_in_context@k": 0.44448360944659854
    }
```

The third output file is `${evaluation_dir}/${retriever}/${dataset}_results_by_k.jpg` which plots retriever performance by k. 

After evaluating each retriever as above, you can use `evaluate_retriever.ipynb` to compare different retrievers across values of k.








