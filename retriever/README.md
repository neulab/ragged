We support retrievers BM25 and ColBERT and retrieval corpuses kilt_wikipedia (KILT version) and the 2023 Annual Medline Corpus

There are 3 steps: 
1. Get top-k passages from the retrievers
2. Evaluate each retriever separately
3. Compare retrievers

# 1. Get top-k passages from the retrievers
## BM25
This involves two steps: 1) using pyserini repo to get BM25 indices, 2) using KILT repo to get BM25 outputs. 
<!-- 1. Clone [pyserini repo](https://github.com/castorini/pyserini). -->

1. Install Java 11.

2. Run [BM25/get_indices.sh](https://github.com/neulab/ragged/blob/main/retriever/BM25/get_indices.sh). This will output a folder `${index_dir}/${corpus}_jsonl`. 

<!-- 3. Clone [KILT repo](https://github.com/facebookresearch/KILT/tree/main). -->

4. Replace `${index_dir}/${corpus}_jsonl` in [`BM25/default_bm25.json`](https://github.com/neulab/ragged/blob/main/retriever/BM25/default_bm25.json).

<!-- 5. Replace `KILT/kilt/retrievers/BM25_connector.py` with [`BM25/BM25_connector.py`](https://github.com/neulab/ragged/blob/main/retriever/BM25/BM25_connector.py). -->

3. Customize `KILT/kilt/configs/${dataset}.json` for your select dataset.

4. Run [`BM25/bm25.sh`](https://github.com/neulab/ragged/blob/main/retriever/BM25/bm25.sh) to output the top-k passages for each query in the file`${prediction_dir}/bm25/${dataset}.jsonl`.

Each line corresponds to a query. This is an example of one line:
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

## ColBERT
1. Clone our [modified version](https://github.com/jenhsia/RAGGED_ColBERT/tree/merged) of the [original ColBERT repo](https://github.com/stanford-futuredata/ColBERT).

2. Download the [pre-trained ColBERTv2](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#:~:text=pre%2Dtrained%20ColBERTv2%20checkpoint) checkpoint into your $model_dir. This checkpoint has been trained on the MS MARCO Passage Ranking task. You can also optionally [train your own ColBERT model](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#:~:text=train%20your%20own%20ColBERT%20model).

3. Run [`ColBERT/colbert.sh`](https://github.com/neulab/ragged/blob/main/retriever/ColBERT/colbert.sh) to output `${prediction_dir}/colbert/${dataset}.jsonl`.

# 2. Evaluate each retriever
To evaluate the predictions, run [`evaluate_retriver.sh`](https://github.com/neulab/ragged/blob/main/retriever/evaluate_retriever.sh) with the appropriate arguments.
This outputs 3 files.

The first output file is `${evaluation_dir}/${retriever}/${dataset}.jsonl`. 
<!-- For each line/query, we include paragraph-level results for each of the k retrieved paragraphs. We include an example of one line below, 
where 'id' corresponds to the query id. -->
Each line corresponds to a query. This is an example of one line:
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

# 3. Compare retrievers
After evaluating each retriever separately as above, use [`evaluate_retriever.ipynb`](https://github.com/neulab/ragged/blob/main/retriever/evaluate_retriever.ipynb) to compare different retrievers across values of k.








