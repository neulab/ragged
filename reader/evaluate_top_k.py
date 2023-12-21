from readers.utils import combine_all_files
from evaluation.eval_downstream import _bertscore, _exact_match_score, _f1_score, _metric_max_over_ground_truths, _rougel_score, get_gold_answers, normalize_answer
from file_utils import BASE_FOLDER, READER_BASE_FOLDER, load_jsonl, save_json, save_jsonl
import pandas as pd

def do_eval_like_kilt(guess_answer, gold_candidate_answers):
    # 0. accuracy = strict exact match
    local_accuracy = 0
    if guess_answer in gold_candidate_answers:
        local_accuracy = 1

    # 1. normalized exact match
    local_em = _metric_max_over_ground_truths(
        _exact_match_score, guess_answer, gold_candidate_answers
    )

    substring_match = False
    for g in gold_candidate_answers:
        g_new = normalize_answer(g)
        normalize_guess_answer = normalize_answer(guess_answer)
        if normalize_guess_answer in g_new or g_new in normalize_guess_answer:
            substring_match = True
            break


    # 2. normalized f1
    local_f1 = _metric_max_over_ground_truths(
        _f1_score, guess_answer, gold_candidate_answers
    )

    # 3. rougel
    local_rougel = _metric_max_over_ground_truths(
        _rougel_score, guess_answer, gold_candidate_answers
    )

    local_bertscore = _metric_max_over_ground_truths(
        _bertscore, guess_answer, gold_candidate_answers
    )
    

    return local_accuracy, local_em, substring_match, local_f1, local_rougel, local_bertscore

def evaluate_reader_results(reader_output, gold_data):

    gold_data_id_map = {str(gd["id"]):gd for gd in gold_data}

    print(len(gold_data))
    print(len(reader_output))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_substring_match = 0
    normalized_f1 = 0
    rougel = 0
    bertscore_f1=0
    bertscore_p=0
    bertscore_r=0

    for reader_output_info in reader_output:
        total_count+=1

        guess_answer = reader_output_info["answer"]

        gold_data_point = gold_data_id_map[reader_output_info["id"]]
        gold_candidate_answers = [x["answer"] for x in gold_data_point["output"] if "answer" in x] #considering only the short answers
        reader_output_info["gold_answers"] = gold_candidate_answers
        
        local_accuracy, local_em, substring_match, local_f1, local_rougel, local_bertscore = do_eval_like_kilt(guess_answer, gold_candidate_answers)

        accuracy+=local_accuracy
        normalized_em+=local_em
        normalized_substring_match+=substring_match
        normalized_f1+=local_f1
        rougel+=local_f1
        bertscore_f1+=local_bertscore["bertscore_f1"]
        bertscore_p+=local_bertscore["bertscore_precision"]
        bertscore_r+=local_bertscore["bertscore_recall"]

        reader_output_info["answer_evaluation"] = {
                "accuracy":local_accuracy,
                "exact_match": local_em,
                "substring_match":substring_match,
                "f1":local_f1,
                "rougel":local_rougel,
                "bertscore": local_bertscore
            }
        
    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_substring_match /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        bertscore_f1 /= total_count
        bertscore_p /= total_count
        bertscore_r /= total_count

    method_metrics = {
        "accuracy": round(accuracy, 4),
        "exact_match": round(normalized_em, 4),
        "substring_match":round(normalized_substring_match, 4),
        "f1": round(normalized_f1, 4),
        "rougel": round(rougel, 4),
        "bertscore_f1": round(bertscore_f1, 4),
        "bertscore_p": round(bertscore_p, 4),
        "bertscore_r": round(bertscore_r, 4)
    }

    print(f"total questions - dev: {total_count}/2837")
    print("Reader metrics : ", method_metrics)
    
    return reader_output,method_metrics

if __name__ == "__main__":
    
    # root_dirs = ["/data/user_data/jhsia2/dbqa/reader_results/llama_70b", "/data/user_data/jhsia2/dbqa/reader_results/flanT5"]
    # root_dirs = ["/data/user_data/jhsia2/dbqa/reader_results/llama_7b", "/data/user_data/jhsia2/dbqa/reader_results/flanUl2"]
    root_dirs = [f"{READER_BASE_FOLDER}/llama_7b"]
    retriever_path_map = {
        "bm25": f"{BASE_FOLDER}/retriever_results/predictions/bm25/",
        "colbert": f"{BASE_FOLDER}/retriever_results/predictions/colbert/"

    }

    dataset_map = {
        "hotpotqa" : "/data/user_data/afreens/kilt/gold_data/hotpotqa-dev-kilt.jsonl",
        "nq": "/data/user_data/afreens/kilt/gold_data/nq-dev-kilt.jsonl"
    }
    for basedir in root_dirs:
        for retriever in ["bm25"]:
            for dataset in ["hotpotqa"]:
                root_dir = f"{basedir}/{dataset}/{retriever}/"
    
    gold_file = dataset_map[dataset]

    top_ks= [ "baseline", "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]
    # top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]
    metrics_map = {}
    metrics_save_path = f"{root_dir}combined_metrics.json"
    for top_k in top_ks:
        print(top_k)
        base_folder = f"{root_dir}{top_k}/"
        evaluation_file_path = f"{base_folder}all_data_evaluated.jsonl"
        all_data = combine_all_files(base_folder, f"{base_folder}all_data.jsonl")
        gold_data = load_jsonl(gold_file)

        all_data, metrics = evaluate_reader_results(all_data, gold_data)
        metrics_map[top_k] = metrics
        save_json(metrics_map, metrics_save_path)
        save_jsonl(all_data, evaluation_file_path)

    import pandas as pd
    df = pd.DataFrame(metrics_map)
    df.T.to_csv(metrics_save_path[:-4]+"csv")
    print(df.T)








