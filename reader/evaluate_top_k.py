from rouge import Rouge
from file_utils import load_jsonl, save_json, save_jsonl
import pandas as pd
import os
from utils import DATA_FOLDER, READER_FOLDER, dataset_map
from word2number import w2n
import argparse
import pandas as pd
import re
import argparse
import re
import string
from rouge import Rouge
from evaluate import load
bertscore = load("bertscore")

from collections import Counter

# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    # pdb.set_trace()
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    if not isinstance(scores_for_ground_truths[0], dict):
        return max(scores_for_ground_truths)
    else:
        return max(scores_for_ground_truths, key=lambda x:x["bertscore_f1"])


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _bertscore(prediction, ground_truth):
    results = bertscore.compute(predictions=[normalize_answer(prediction)], references=[normalize_answer(ground_truth)], lang="en")
    return {
        "bertscore_precision" : results["precision"][0],
        "bertscore_recall" : results["recall"][0],
        "bertscore_f1" : results["f1"][0]
    }

# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def is_potential_number(word):
    """
    Check if a word is a potential part of a number in textual form.
    """
    number_parts = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
                    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", 
                    "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion"]
    return word.lower() in number_parts

def convert_textual_numbers_to_numeric(sentence):
    # print()
    # print(sentence)
    """
    Convert textual numbers within a sentence to numeric form, handling compound numbers.
    Args:
    - sentence (str): The sentence to process.
    
    Returns:
    - str: The sentence with numbers converted to numeric form.
    """
    words = sentence.split()
    converted_words = []
    current_number_phrase = []

    for word in words:
        if is_potential_number(word):
            current_number_phrase.append(word)
        else:
            if current_number_phrase:
                # Convert the current number phrase to a number
                number_string = " ".join(current_number_phrase)
                try:
                    numeric_value = w2n.word_to_num(number_string)
                    converted_words.append(str(numeric_value))
                except ValueError:
                    # If conversion fails, keep the original phrase
                    converted_words.extend(current_number_phrase)
                current_number_phrase = []
            
            converted_words.append(word)

    # Handle any remaining number phrase at the end
    if current_number_phrase:
        try:
            number_string = " ".join(current_number_phrase)
            numeric_value = w2n.word_to_num(number_string)
            converted_words.append(str(numeric_value))
        except ValueError:
            converted_words.extend(current_number_phrase)

    return ' '.join(converted_words)

def do_eval_like_kilt(guess_answer, gold_candidate_answers, eval_info, WITH_BERT):
    if not eval_info:
        eval_info = {}

    guess_answer = convert_textual_numbers_to_numeric(guess_answer)
    gold_candidate_answers = [convert_textual_numbers_to_numeric(ans) for ans in gold_candidate_answers]

    if "accuracy" not in eval_info:
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
    else:
        local_accuracy = eval_info["accuracy"]

    if "exact_match" not in eval_info:
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
    else:
        local_em = eval_info["exact_match"]

    if "substring_match" not in eval_info:
        substring_match = False
        for g in gold_candidate_answers:
            g_new = normalize_answer(g)
            normalize_guess_answer = normalize_answer(guess_answer)
            if normalize_guess_answer in g_new or g_new in normalize_guess_answer:
                substring_match = True
                break
    else:
        substring_match = eval_info["substring_match"]

    if "f1" not in eval_info:
        local_f1 = _metric_max_over_ground_truths(
        _f1_score, guess_answer, gold_candidate_answers
        )
    else:
        local_f1 = eval_info["f1"]

    if "rougel" not in eval_info:
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
    else:
        local_rougel = eval_info["rougel"]

    
    local_bertscore = None
    if WITH_BERT:
        if "bertscore" not in eval_info:
            local_bertscore = _metric_max_over_ground_truths(
                _bertscore, guess_answer, gold_candidate_answers
            )
        else:
            local_bertscore = eval_info["bertscore"]
    
    return local_accuracy, local_em, substring_match, local_f1, local_rougel, local_bertscore

def evaluate_reader_results(reader_output, gold_data, WITH_BERT, args):

    gold_data_id_map = {str(gd["id"]):gd for gd in gold_data}

    print('gold data len', len(gold_data))
    print('reader out len', len(reader_output))
    if len(gold_data) != len(reader_output):
        print('size mismatch')

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_substring_match = 0
    normalized_f1 = 0
    rougel = 0
    if WITH_BERT:
        bertscore_f1=0
        bertscore_p=0
        bertscore_r=0

    for reader_output_info in reader_output:
        total_count+=1

        guess_answer = reader_output_info["answer"]

        gold_data_point = gold_data_id_map[reader_output_info["id"]]
        gold_candidate_answers = [x["answer"] for x in gold_data_point["output"] if "answer" in x] #considering only the short answers
        if args.merge_list_answers and gold_data_point.get("question_type", "")=="list":
            # print("merge_list_answers")
            gold_candidate_answers = [" ".join(gold_candidate_answers)]
        reader_output_info["gold_answers"] = gold_candidate_answers
        
        local_accuracy, local_em, substring_match, local_f1, local_rougel, local_bertscore = do_eval_like_kilt(guess_answer, gold_candidate_answers, reader_output_info.get("answer_evaluation"), WITH_BERT)

        accuracy+=local_accuracy
        normalized_em+=local_em
        normalized_substring_match+=substring_match
        normalized_f1+=local_f1
        rougel+=local_f1
        if WITH_BERT:
            bertscore_f1+=local_bertscore["bertscore_f1"]
            bertscore_p+=local_bertscore["bertscore_precision"]
            bertscore_r+=local_bertscore["bertscore_recall"]

        reader_output_info["answer_evaluation"] = {
                "accuracy":local_accuracy,
                "exact_match": local_em,
                "substring_match":substring_match,
                "f1":local_f1,
                "rougel":local_rougel
            }
        if WITH_BERT:
            reader_output_info["answer_evaluation"]["bertscore"] = local_bertscore
        
    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_substring_match /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        if WITH_BERT:
            bertscore_f1 /= total_count
            bertscore_p /= total_count
            bertscore_r /= total_count

    method_metrics = {
        "accuracy": round(accuracy, 4),
        "exact_match": round(normalized_em, 4),
        "substring_match":round(normalized_substring_match, 4),
        "f1": round(normalized_f1, 4),
        "rougel": round(rougel, 4)
    }
    if WITH_BERT:
        method_metrics["bertscore_f1"] = round(bertscore_f1, 4)
        method_metrics["bertscore_p"] = round(bertscore_p, 4)
        method_metrics["bertscore_r"] = round(bertscore_r, 4)

    print(f"total questions - dev: {total_count}/{len(gold_data)}")
    print("Reader metrics : ", method_metrics)
    
    return reader_output,method_metrics

def baseline_evaluation(model, dataset, with_bert=False, args=None):
    #gold baseline evaluation
    result_dir = os.path.join(READER_FOLDER, args.model, args.dataset, args.retriever)
    print(model, dataset)

    gold_data = load_jsonl(os.path.join(DATA_FOLDER, dataset_map[dataset]))
    reader_output = load_jsonl(os.path.join(result_dir, "reader_results.jsonl"))
    evaluated_data, metrics = evaluate_reader_results(reader_output, gold_data, with_bert, args)
    
    metrics_save_path = os.path.join(result_dir, "combined_metrics.json")
    save_json(metrics, metrics_save_path)

    evaluation_file_path = os.path.join(result_dir, "all_data_evaluated.jsonl")
    save_jsonl(evaluated_data, evaluation_file_path)

def top_k_evaluation(model, retriever, dataset, with_bert=False, args=None):

    k_list_str = re.split(r',\s*', args.k_list)
    k_list = [int(num) for num in k_list_str]

    print(f"Eval - {model}/{dataset}/{retriever}/")
    root_dir = os.path.join(READER_FOLDER, model, dataset, retriever, args.retrieval_mode)
    gold_file = os.path.join(DATA_FOLDER, dataset_map[dataset])

    metrics_map = {}
    metrics_save_path = os.path.join(root_dir, 'combined_metrics.json')
    for k in k_list:
        k_dir = os.path.join(root_dir, f'top_{k}')
        reader_output = load_jsonl(os.path.join(k_dir, "reader_results.jsonl"))
        gold_data = load_jsonl(gold_file)

        evaluated_data, metrics = evaluate_reader_results(reader_output, gold_data, with_bert, args)
        metrics_map[f'top_{k}'] = metrics
        save_json(metrics_map, metrics_save_path)
        evaluation_file_path = os.path.join(k_dir, "all_data_evaluated.jsonl")
        save_jsonl(evaluated_data, evaluation_file_path)

    df = pd.DataFrame(metrics_map)
    df.T.to_csv(metrics_save_path[:-4]+"csv")
    print(df.T)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=str, help="Name of reader model")
    parser.add_argument("--retriever", type=str, default=[], help="Name of retriever")
    parser.add_argument("--dataset", type=str, help="Name of dataset")
    parser.add_argument('--with_bert', dest='with_bert', action='store_true', help="(Optional) Flag to include BERTScore in the evaluation metrics. Note this slows down reader evaluation significantly.")
    parser.add_argument("--retrieval_mode", help = "One of 3 context choices: provide all top-k retrieved passages (top_k), provide only the passages marked as relevant within the top-k retrieved passages(top_positive), provide only the passages not marked as relevant within teh top-k retrieved passages (top_negative)")
    parser.add_argument('--merge_list_answers', dest='merge_list_answers', action='store_true', help="Flag to merge list-type answers for evaluation. (this flag will be used only when the data under reader_results.jsonl has 'question_type' attribute set to 'list')")
    parser.add_argument("--k_list", help = 'Comma-separated list of k (number of passages)')
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.retriever == 'gold' or args.retriever == 'no_context':
        baseline_evaluation(args.reader, args.dataset, with_bert=args.with_bert, args=args)
    else:
        top_k_evaluation(args.reader, args.retriever, args.dataset, with_bert=args.with_bert, args=args)


