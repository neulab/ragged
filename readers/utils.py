import glob
import os
from file_utils import load_data, store_data, write_json

CONTEXT_PROMPT = "Give simple short one phrase answers for the questions based on the context"
NO_CONTEXT_PROMPT = "Give simple short one phrase answers for the question"
def create_prompt(question, context):
    if context:
        return f"{CONTEXT_PROMPT}\nContext: {context}\nQuestion: {question}\nAnswer: ".strip()
    else:
        return f"{NO_CONTEXT_PROMPT}\nQuestion: {question}\nAnswer: ".strip()
    

def convert_reader_output_to_zeno_format(reader_output_file, retriever_eval_file, top_k=1):
    reader_output_data = load_data(reader_output_file, sort_by_id=True)
    retriever_eval_data = load_data(retriever_eval_file, sort_by_id=True)

    assert [x['id'] for x in reader_output_data] == [x['id'] for x in retriever_eval_data]
    

    zeno_format_data = []
    for reader_q_info, retriever_info in zip(reader_output_data, retriever_eval_data):
        assert reader_q_info["id"] == retriever_info["id"]
        qid = reader_q_info["id"]
        question = reader_q_info["input"]
        answer = reader_q_info["output"]["provenance"][0]["answer"]
        answer_evaluation = reader_q_info["output"]["provenance"][0]["answer_evaluation"]
        retrieved_passages = []
        for retrieved_passage_info, retrieved_passage_info_eval in zip(reader_q_info["output"]["provenance"][:top_k], retriever_info["doc-level results"][:top_k]):
            assert retrieved_passage_info["docid"] == retrieved_passage_info_eval["wiki_par_id"]
            retrieved_passages.append({
                "reference":retrieved_passage_info["docid"],
                "retrieval_evaluation": retrieved_passage_info_eval,
                "text" : retrieved_passage_info["text"],
                "score": retrieved_passage_info["score"]
            })
            
        
        
        zeno_format_data.append({
            "data": question,
            "output":[{
                "answer" : answer,
                "answer_evaluation": answer_evaluation,
                "retrieved" : retrieved_passages
            }]
        })
    assert (len(zeno_format_data) ==  len(reader_output_data))
    return zeno_format_data


def convert_gold_data_to_zeno_format(gold_file):
    zeno_format_data = []
    gold_data = load_data(gold_file)
    for q_info in gold_data:
        qid = q_info["id"]
        question = q_info["input"]
        answers = q_info["output"]
        outputs = []
        for answer in answers:
            if "answer" not in answers or "provenance" not in answers:
                continue
            outputs.append({
                "answer": answer["answer"],
                "retrieved": [get_document_text(p["wikipedia_id"])]
            })
            retrieved_passage_info = q_info["output"]["provenance"][0]
            retrieved_passages = [
                {
                    "reference":retrieved_passage_info["docid"],
                    "text" : retrieved_passage_info["text"],
                    "score": retrieved_passage_info["score"]                
                }
            ]
        zeno_format_data.append({
            "data": question,
            "output":{
                "answer" : answer,
                "retrieved" : retrieved_passages
            }
        })
    assert (len(zeno_format_data) ==  len(gold_data))
    return zeno_format_data


def zeno_format_conversion_starter():
    reader_output_file = "/data/user_data/jhsia2/dbqa/reader_results/reader_output_evaluated_with_baselines.jsonl"
    retriever_eval_file = "/data/user_data/jhsia2/dbqa/retriever_results/evaluations/bm25/nq-dev-kilt.jsonl"
    zeno_format_reader_data = convert_reader_output_to_zeno_format(reader_output_file, retriever_eval_file)

    write_json(zeno_format_reader_data, "/data/user_data/jhsia2/dbqa/reader_results/top1_reader_results_zeno.json")
    write_json(zeno_format_reader_data, "/home/afreens/top1_reader_results_zeno.json")


    # gold_data_file = "/data/user_data/afreens/kilt/gold_data/nq-dev-kilt.jsonl"
    # zeno_format_gold_data = convert_gold_data_to_zeno_format(gold_data_file)

    # write_json(zeno_format_gold_data, "/data/user_data/jhsia2/dbqa/reader_results/gold_reader_results_zeno.json")
    # write_json(zeno_format_gold_data, "/home/afreens/gold_reader_results_zeno.json")

def combine_all_files(base_path, output_path=None):
    all_data = []
    all_data_unique = []
    if os.path.exists(f"/{base_path.strip('/')}/all_data.jsonl"):
        return load_data(f"/{base_path.strip('/')}/all_data.jsonl")
    for file in glob.glob(f"/{base_path.strip('/')}/*"):
        if not file.endswith(".jsonl") or "error" in file:
            continue
        all_data.extend(load_data(file))

    qids = set()
    for x in all_data:
        if x["id"] in qids:
            continue
        qids.add(x["id"])
        all_data_unique.append(x)
    print(len(all_data_unique))
    # assert len(all_data_unique) == 2837
    
    if output_path:    
        store_data(output_path, all_data_unique)
    return all_data_unique

    

# if __name__ == "__main__":
#     combine_all_files("/data/user_data/afreens/kilt/llama/top2", "/data/user_data/afreens/kilt/llama/reader_top2_results.json")
#     combine_all_files("/data/user_data/afreens/kilt/llama/top3", "/data/user_data/afreens/kilt/llama/reader_top3_results.json")



    

