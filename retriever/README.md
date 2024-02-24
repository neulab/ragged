We support retrievers BM25 and ColBERT and retrieval corpuses Wikipedia (KILT version) and the 2023 Annual Medline Corpus

For each, you need download the repo and process the data differently.

Download and process corpus data

    Download BioAS corpus - Pubmed

        python download_pubmed_corpus.py --data_dir /data/tir/projects/tir6/general/afreens/dbqa/data
        This downloads the pubmed corpus in unprocessed form to ${data_dir}/bioasq/annual_zips/

        use python create_pubmed_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
        This outputs 'pubmed/pubmed_jsonl/pubmed.jsonl' and 'pubmed/id2title.json' in corpus_dir 

    Download Wiki corpus
        python create_wiki_paragraph_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
        This outputs 'wikipedia/wikipedia_jsonl/wikipedia.jsonl" in your corpus_dir

    To process corpus for ColBERT format , use
        create_corpus_tsv.py --corpus_name wikipedia --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/
        This outputs '${corpus_name}/${corpus_name}.tsv' in corpus_dir

Download query dataset
    Download NQ, hotpotqa from KILT repo

    Download BioASQ
        From Bioasq website, download the following into data_dir/bioasq/
        Task11BGoldenEnriched/11B*_golden.json and BioASQ-training11b/training11b.json from BioASQ
    
        python compile_bioasq_questions.py --data_dir --corpus_dir 
        This outputs bioasq.jsonl in the data_dir

    To process query tsv for ColBERT, use
        python create_query_tsv.py --data_dir --dataset
        This outputs a {dataset}-queries.tsv

Use BM25 for predictions
    You should download the pyserini repo and KILT repo. https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation.

    Run bm25.sh

Use ColBERT for predictions
    Run colbert.sh





