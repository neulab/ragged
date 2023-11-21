


import csv
from file_utils import load_data, read_json, write_json


def convert_reader_results_to_zeno(reader_output_data, retriever_eval_data):
    # print([(x[0], x[1]) for x in zip([d["id"] for d in reader_output_data], [d["id"] for d in retriever_eval_data])])
    assert [d["id"] for d in reader_output_data] == [d["id"] for d in retriever_eval_data]

    zeno_format_data = []
    for reader_q_info, retriever_info in zip(reader_output_data, retriever_eval_data):
        assert reader_q_info["id"] == retriever_info["id"]
        qid = reader_q_info["id"]
        question = reader_q_info["input"]
        answer = reader_q_info["answer"]
        answer_evaluation = reader_q_info["answer_evaluation"]
        for retrieved_passage_info, retrieved_passage_info_eval in zip(reader_q_info["retrieved_passages"], retriever_info["doc-level results"][:len(reader_q_info["retrieved_passages"])]):
            assert retrieved_passage_info["docid"] == retrieved_passage_info_eval["wiki_par_id"]
            retrieved_passage_info.update(retrieved_passage_info_eval)
            
        zeno_format_data.append({
            "data": question,
            "output":[{
                "answer" : answer,
                "answer_evaluation": answer_evaluation,
                "retrieved" : reader_q_info["retrieved_passages"]
            }],
            "gold_answers": reader_q_info["gold_answers"],
            "summary context evaluation": {
                "wiki_id_match": any([r["wiki_id_match"] for r in reader_q_info["retrieved_passages"]]),
                "wiki_par_id_match": any([r["wiki_par_id_match"] for r in reader_q_info["retrieved_passages"]])
            }
        })
    assert (len(zeno_format_data) ==  len(reader_output_data))
    return zeno_format_data

def convert_gold_to_zeno():
    wiki_par_ids_data = read_json("/data/user_data/jhsia2/dbqa/data/gold-nq-dev-kilt.json")
    gold_data = load_data("/data/user_data/jhsia2/dbqa/data/nq-dev-kilt.jsonl")
    tsv_file = "/data/user_data/jhsia2/dbqa/data/kilt_knowledgesource.tsv"

    par_id_to_text_map = {}

    with open("/data/user_data/jhsia2/dbqa/data/kilt_knowledgesource.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            print(i)
            par_id_to_text_map[row[0]] = row[1]
            # break
    zeno_format_data = []
    for ques_info in gold_data:
        qid = ques_info["id"]
        print(qid)
        question = ques_info["input"]
        answers = ques_info["output"]
        output_answers = []
        for answer in answers:
            if "answer" not in answer or "provenance" not in answer:
                continue
            for p in answer["provenance"]:
                p["text"] = par_id_to_text_map[str(p["wikipedia_id"])+"_"+str(p["start_paragraph_id"]+1)]

            output_answers.append({
                "answer": answer["answer"],
                "retrieved": answer["provenance"]
            })
            
        zeno_format_data.append({
            "data": question,
            "output":output_answers
        })
    write_json(zeno_format_data, "/data/user_data/jhsia2/dbqa/data/gold_nq_zeno_file.json")
if __name__ == "__main__":

    root_dir = "/data/user_data/afreens/kilt/llama/hotpot/bm25/"

    dataset = "hotpot"
    retriever_eval_file_map = {
        "hotpot" : "/data/user_data/jhsia2/dbqa/retriever_results/evaluations/bm25/hotpotqa-dev-kilt.jsonl",
        "nq" : "/data/user_data/jhsia2/dbqa/retriever_results/evaluations/bm25/nq-dev-kilt.jsonl"
    }
    retriever_eval_file = retriever_eval_file_map[dataset]
    top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20", "top30", "top50"]

    for top_k in top_ks:
        print(top_k)
        base_folder = f"{root_dir}/{top_k}/"
        evaluation_file_path = f"{base_folder}all_data_evaluated.jsonl"

        reader_output_data = load_data(evaluation_file_path)
        retriever_eval_data = load_data(retriever_eval_file)
        
        zeno_format_data = convert_reader_results_to_zeno(reader_output_data, retriever_eval_data)
        write_json(zeno_format_data, f"{base_folder}reader_results_zeno.json")

    # convert_gold_to_zeno()