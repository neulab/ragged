from readers.utils import combine_all_files
from evaluation.eval_downstream import _exact_match_score, _f1_score, _metric_max_over_ground_truths, _rougel_score, get_gold_answers, normalize_answer
from file_utils import load_data, store_data, write_json

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
    

    return local_accuracy, local_em, substring_match, local_f1, local_rougel

def evaluate_reader_results(reader_output, gold_data):

    gold_data_id_map = {gd["id"]:gd for gd in gold_data}

    print(len(gold_data))
    print(len(reader_output))

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_substring_match = 0
    normalized_f1 = 0
    rougel = 0

    for reader_output_info in reader_output:
        total_count+=1

        guess_answer = reader_output_info["answer"]

        gold_data_point = gold_data_id_map[reader_output_info["id"]]
        gold_candidate_answers = [x["answer"] for x in gold_data_point["output"] if "answer" in x] #considering only the short answers
        reader_output_info["gold_answers"] = gold_candidate_answers
        
        local_accuracy, local_em, substring_match, local_f1, local_rougel = do_eval_like_kilt(guess_answer, gold_candidate_answers)

        accuracy+=local_accuracy
        normalized_em+=local_em
        normalized_substring_match+=substring_match
        normalized_f1+=local_f1
        rougel+=local_f1

        reader_output_info["answer_evaluation"] = {
                "accuracy":local_accuracy,
                "exact_match": local_em,
                "substring_match":substring_match,
                "f1":local_f1,
                "rougel":local_rougel
            }
        
    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_substring_match /= total_count
        normalized_f1 /= total_count
        rougel /= total_count

    method_metrics = {
        "accuracy": round(accuracy, 4),
        "exact_match": round(normalized_em, 4),
        "substring_match":round(normalized_substring_match, 4),
        "f1": round(normalized_f1, 4),
        "rougel": round(rougel, 4),
    }

    print(f"total questions - dev: {total_count}/2837")
    print("Reader metrics : ", method_metrics)
    
    return reader_output,method_metrics

if __name__ == "__main__":
    top_ks= ["baseline", "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]
    metrics_map = {}
    metrics_save_path = "/data/user_data/afreens/kilt/flanT5/nq/exp2/combined_metrics.json"
    for top_k in top_ks:
        print(top_k)
        base_folder = f"/data/user_data/afreens/kilt/flanT5/nq/exp2/{top_k}/"
        # gold_file = "/data/user_data/afreens/kilt/gold_data/hotpotqa-dev-kilt.jsonl"
        gold_file = "/data/user_data/afreens/kilt/gold_data/nq-dev-kilt.jsonl"
        evaluation_file_path = f"{base_folder}all_data_evaluated.jsonl"

        all_data = combine_all_files(base_folder, f"{base_folder}all_data.jsonl")
        gold_data = load_data(gold_file)

        all_data, metrics = evaluate_reader_results(all_data, gold_data)
        metrics_map[top_k] = metrics
        write_json(metrics_map, metrics_save_path)
        store_data(evaluation_file_path, all_data)

    import pandas as pd
    df = pd.DataFrame(metrics_map)
    df.T.to_csv(metrics_save_path[:-4]+"csv")
    print(df.T)








