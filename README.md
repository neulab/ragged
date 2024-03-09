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
'conda create --name ragged_env --file requirements.txt'.

To run and evaluate the retriever, see retriver/README.md

To run and evaluate the reader, see reader/README.md

To conduct downstream RAGGED analysis, see analysis_framework/README.md


## Datasets
To download the datasets used in the paper, see instructions in reader/README.md and folder X. 


## Citation
If you use our code, datasets, or concepts from our paper in your research, we would appreciate citing it in your work. Here is an example BibTeX entry for citing our paper:
```bibtex
@article{x,
  title={x},
  author={x},
  journal={x},
  year={x}
}
```
## Contact
For any questions, feedback, or discussions regarding this project, please feel free to contact us:

- **Contact**
  - Email: [jhsia2@cs.cmu.edu](mailto:jhsia2@cs.cmu.edu), [afreens@cs.cmu.edu](mailto:afreens@cs.cmu.edu)

