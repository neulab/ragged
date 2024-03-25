<!-- # ragged -->
<!-- https://hub.zenoml.com/project/jhsia2/Document%20QA

Combined (deprecated): https://hub.zenoml.com/project/84123876-66c1-46b5-9844-28c5828b340a/Document%20QA 

Natural Questions: https://hub.zenoml.com/project/aed6ce66-ee8b-4d94-997c-8092d031e6aa/Document%20QA%20-%20nq/explore

HotpotQA: https://hub.zenoml.com/project/a8ddbb03-a920-4376-80c0-0999d66bb540/Document%20QA%20-%20hotpotqa/explore

Bioasq (complete medline corpus): https://hub.zenoml.com/project/17d95f38-aa53-4eb3-818e-385ae2d37785/Document%20QA%20-%20complete_bioasq/explore

Bioasq (sampled): https://hub.zenoml.com/project/e7a27fce-bf84-4f52-ac1b-3d7975c44bf4/Document%20QA%20-%20bioasq/explore -->

# [RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems](arxivlink)

## Description
Retrieval-augmented generation (RAG) greatly benefits language models (LMs) by providing additional context for tasks such as document-based question answering (DBQA). 
Despite its potential, the power of RAG is highly dependent on its configuration, raising the question: *What is the optimal RAG configuration?*
To answer this, we introduce the RAGGED framework to analyze and optimize RAG systems. On the representative DBQA tasks, we study two classic sparse and dense retrievers, and four top-performing LMs in encoder-decoder and decoder-only architectures.
Through RAGGED, we uncover that *different models suit substantially varied RAG setups*.
While encoder-decoder models monotonically improve with more documents, we find decoder-only models can only effectively use <5 documents, despite often having a longer context window.
RAGGED offers further insights into LMs' context utilization habits, where we find encoder-decoder models rely more on contexts and are thus more sensitive to retrieval quality, while decoder-only models tend to rely on knowledge memorized during training.

## Installation
To recreate the conda environment, run 
`conda create -n ragged -y python=3.10`
`pip install -r requirements.txt`

To run and evaluate the retriever, see [`retriver/README.md`](https://github.com/neulab/ragged/blob/main/retriever/README.md).

To run and evaluate the reader, see [`reader/README.md`](https://github.com/neulab/ragged/blob/main/reader/README.md).

To conduct downstream RAGGED analysis, see [`analysis_framework/README.md`](https://github.com/neulab/ragged/blob/main/analysis_framework/README.md).


## Datasets
Our datasets are available [on Huggingface](https://huggingface.co/datasets/jenhsia/ragged) 
<!-- To download the datasets used in the paper, see instructions in [`retriver/README.md`](https://github.com/neulab/ragged/blob/main/retriever/README.md). -->
### 1. Download and process corpus datasets

<!-- ## Download BioAS corpus - Pubmed -->
<!-- 
        python download_pubmed_corpus.py --data_dir /data/tir/projects/tir6/general/afreens/dbqa/data
        This downloads the pubmed corpus in unprocessed form to ${data_dir}/bioasq/annual_zips/

        use python create_pubmed_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
        This outputs 'pubmed/pubmed_jsonl/pubmed.jsonl' and 'pubmed/id2title.json' in corpus_dir 
        python create_page_paragraph_jsonl.py --corpus_dir /data/tir/projects/tir6/general/afreens/dbqa/data/corpus_files
    This outputs 'kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl" in your corpus_dir -->

<!-- We provide our datasets from [our Huggingface link](https://huggingface.co/datasets/jenhsia/ragged). -->


<!-- in jsonl/csv format, use [to_json] and ([to_csv](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset)) as follows: -->

Specify `corpus_dir` and `corpus_name` and see `download_data.py` for how to download and save the files in appropriate folders.

For Pubmed corpus for BioASQ, the corpus name is `pubmed`.
<!-- , and the files should be downloaded to `${corpus_dir}/pubmed/pubmed_jsonl/pubmed.jsonl` and `${corpus_dir}/pubmed/id2title.csv`. -->
<!-- `${corpus_dir}/pubmed/pubmed_jsonl/pubmed.jsonl` and `pubmed_id2title`to `${corpus_dir}/pubmed/id2title.csv` -->

<!-- ## Download Wiki corpus -->
For KILT wikipedia corpus, the corpus name is `kilt_wikipedia`.

<!-- , and the files should be downloaded to `${corpus_dir}/kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl` and `${corpus_dir}/kilt_wikipedia/id2title.json`. --> 
<!-- to  `${corpus_dir}/kilt_wikipedia/kilt_wikipedia_jsonl/kilt_wikipedia.jsonl` and `kilt_wikipedia_id2title` to `${corpus_dir}/kilt_wikipedia/id2title.json` -->
  

After downloading the datasets, process the corpus for ColBERT format by running `python retriever/data_processing/create_corpus_tsv.py --corpus $corpus --corpus_dir $corpus_dir`, which outputs `$corpus_dir/${corpus}/${corpus}.json`.

### 2. Download query datasets
Specify `data_dir` and `dataset_name` and see `download_data.py` for how to download the file to `${data_dir}/${dataset_name}.jsonl`.

We support Natural Questions (KILT ver), HotpotQA (KILT ver), and BioASQ11B.
    <!-- Download NQ, hotpotqa from KILT repo as nq.jsonl and hotpotqa.jsonl in the ${data_dir} Download BioASQ
        From Bioasq website, download the following into data_dir/bioasq/
        Task11BGoldenEnriched/11B*_golden.json and BioASQ-training11b/training11b.json from BioASQ
        python compile_bioasq_questions.py --data_dir --corpus_dir 
        This outputs bioasq.jsonl in the data_dir -->

The above files are ready for BM25, but not for ColBERT. To reformat them for ColBERT, run `python retriever/data_processing/create_query_tsv.py --data_dir $data_dir --dataset $dataset`, which outputs `$data_dir/${dataset}-queries.tsv`.

### 3. Adapt your own datasets.
To adapt for BM25, format your corpus and query as jsonl files as instructed [here](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation).
To adapt for ColBERT, format your corpus and query datasets as instructed [here](https://github.com/stanford-futuredata/ColBERT).


## Citation
If you use our code, datasets, or concepts from our paper in your research, we would appreciate citing it in your work. Here is an example BibTeX entry for citing our paper:
```bibtex
@article{hsia2024ragged,
  title={RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems},
  author={Jennifer Hsia and Afreen Shaikh and Zhiruo Wang and Graham Neubig},
  journal={arXiv preprint arXiv:2403.09040},
  year={2024}
}
```
## Contact
For any questions, feedback, or discussions regarding this project, please feel free to open an issue on the repository or contact us:

- **Contact**
  - Issues: [https://github.com/neulab/ragged/issues](https://github.com/neulab/ragged/issues)
  - Email: [jhsia2@cs.cmu.edu](mailto:jhsia2@cs.cmu.edu), [afreens@cs.cmu.edu](mailto:afreens@cs.cmu.edu)

