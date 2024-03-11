This README document provides an overview of the Retrieval Augmented Generation (RAG) codebase, focusing on the reader component. The reader component enhances the text generation process by incorporating context from retrieved documents. Important note: During generation, the context of a question-and-context is trimmed based on the --max\_truncation parameter to ensure that the combined length of the instruction, question, and context does not exceed a specified token limit for a machine learning model.

## Prerequisites
Before you begin, ensure you have met the following requirements:
1.  Access to a hosted API endpoint for text generation. Ensure that your text generation inference API is running and accessible.
2.  Necessary Python packages installed (see requirements.txt for details)
    
## READER RESULTS GENERATION
The reader component can be used in two primary ways: as a library in your Python scripts or via the command line interface provided by generate\_top\_k.py.

The output structure is:

READER_FOLDER
│
├── reader_A/
│   ├── dataset_A/
│   │   ├── retriever_A/
│   │   │   └── retrieval_mode (top_k or top_positive or top_negative)
│   │   │       ├── top_1/
│   │   │       ├── ...
│   │   │       └── top_50/
│   │   ├── gold/
│   │   └── no_context/

#### Via code (for simple checks of the setup):
```
from reader.reader_model import Reader
#Initialize the reader
reader = Reader(hosted_api_endpoint="http://<your-api-endpoint>/", tokenizer=<your_tokenizer>)
prompts = [{"question": "What is the capital of France?", "context": "Paris is the capital of France."}]
responses = reader.generate(prompts, max_new_tokens=50, truncate=2000)
```

Via Command Line (for actual generations)
```
python generate_top_k.py --hosted_api_endpoint "<your-api-endpoint>" --model_name "<model-name>" --retriever "<retriever-name>" --dataset "<dataset-name>"
```

### Parameters:

*   \--hosted\_api\_endpoint: The network address where the model server is running, formatted as : `<hostname or IP address>:<port>`.
*   \--k: Specifies the number of top retrieved contexts (passages) to be used by the reader for generating an answer. Default is 1.
*   \--batch\_size: The number of inputs (questions with context) that the reader model processes in one go. Default is 50.
*   \--model\_name: Identifier for the model being used, results for this model are stored under `<base folder for results>/<model_name>`. Base folder for results is specified in [ragged/utils.py](https://github.com/neulab/ragged/blob/main/utils.py)
*   \--retriever: Name of the retriever component used, results for this retriever are stored under `<base folder>/<model_name>/<retriever>`. This could be a retriever model, `no_context`, or `gold` (provide all passages marked as relevant).
*   \--dataset: Name of the dataset being processed, results for this retriever are stored under `<base folder>/<model_name>/<retriever>/<dataset>`
*   \--max\_new\_tokens: Maximum number of new tokens the model is allowed to generate for each answer.
*   \--max\_truncation: The maximum number of tokens from the input (including instruction, context, and question) fed to the reader. Inputs longer than this are truncated.
*   \--retrieval\_mode: One of 3 context choices: provide all top-k retrieved passages (top_k), provide only the passages marked as relevant within the top-k retrieved passages(top_positive), provide only the passages not marked as relevant within the top-k retrieved passages (top_negative).

## READER RESULTS EVALUATION
The evaluation can be conducted in three modes: 
1. Evaluating LLM's baseline answer generation performance when provided with no contexts.
1. Evaluating LLM's baseline answer generation performance when provided with all of the gold contexts.
2. Evaluating LLM's generated answers when provided with various top\_k retrieved documents. 


To evaluate the reader output given no contexts:
`python evaluate_top_k.py --retriever gold --reader flanT5 --dataset nq`

To evaluate the reader output given gold contexts:
```
python ragged/retriever/data_processing/get_gold_contexts.py --prediction_dir $retriever_prediction_dir\
                                                            --corpus_dir $corpus_dir\
                                                            --corpus $corpus\
                                                            --data_dir $data_dir\
                                                            --dataset $dataset
```
`python evaluate_top_k.py --retriever no_context --reader flanT5 --dataset nq`

To evaluate the reader output for various top\_k configurations: 
`python evaluate_top_k.py --retriever no_context --reader flanT5 --dataset nq --retrieval_mode top_k --k_list 1,5,10,20`

#### Parameters:  
<!-- * --readers: Comma-separated list of reader names to evaluate.
* --retrievers: Comma-separated list of retriever names to evaluate.
* --datasets: Comma-separated list of datasets to evaluate. -->
* --reader: Name of reader model
* --retriever: Name of retriever
* --dataset: Name of dataset
*   \--retrieval\_mode: One of 3 context choices: provide all top-k retrieved passages (top_k), provide only the passages marked as relevant within the top-k retrieved passages(top_positive), provide only the passages not marked as relevant within teh top-k retrieved passages (top_negative)
* --k_list: Comma-separated list of k (number of passages)
* --with\_bert: (Optional) Flag to include BERTScore in the evaluation metrics. Note this slows down reader evaluation significantly.
<!-- * --only\_relevant: Evaluate only the relevant passages.
* --only\_non\_relevant: Evaluate only the non-relevant passages. -->
* --merge\_list\_answers: Flag to merge list-type answers for evaluation. (this flag will be used only when the data under `reader_results.jsonl` has "question_type" attribute set to "list")

### Extending the analysis
Models and Tokenizers: Add the new model and tokenizer configurations in utils.py. Ensure the model is supported by the generation and evaluation script by adding its name and corresponding tokenizer to the tokenizer\_map and tokenizer\_path\_map.
