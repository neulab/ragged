import os
import pprint
from file_utils import BASE_FOLDER, NOISY_READER_BASE_FOLDER, ONLY_GOLD_READER_BASE_FOLDER, READER_BASE_FOLDER, load_json, load_jsonl, save_jsonl
from reader.utils import combine_all_files


# # Check if everything is generated

base_folder = ONLY_GOLD_READER_BASE_FOLDER
models = "llama_7b llama_70b flanT5 flanUl2".split(" ")
retrievers = "colbert".split(" ")
datasets = "nq hotpotqa complete_bioasq".split(" ")
print(base_folder)
for model in models:
    # print(f"{model} ------------------------")
    for dataset in datasets:
        top_ks = [ "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"] #if dataset == "complete_bioasq" else [ "top1", "top2", "top3", "top5", "top10" ]
        for retriever in retrievers:
            for top_k in top_ks:
                file = f"{base_folder}/{model}/{dataset}/{retriever}/{top_k}/reader_output_index_0_to_6000.jsonl"
                try:
                    # print(file)
                    data1 = load_jsonl(file)

                    if len(data1) in [2837, 5600, 3837]:
                        # print(f"{model} - {dataset} - {retriever} - {top_k} ---> DONE")
                        pass
                    elif len(data1)>0:
                        print(f"{model} - {dataset} - {retriever} - {top_k} ---> Partial, {len(data1)}")
                    else:
                        print(f"{model} - {dataset} - {retriever} - {top_k} ---> X")
                except Exception:
                    print(f"{model} - {dataset} - {retriever} - {top_k} ---> X")


base_folder = NOISY_READER_BASE_FOLDER
models = "llama_7b llama_70b flanT5 flanUl2".split(" ")
retrievers = "colbert".split(" ")
datasets = "hotpotqa".split(" ")
print(base_folder)
for model in models:
    # print(f"{model} ------------------------")
    for dataset in datasets:
        top_ks = [ "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]
        for retriever in retrievers:
            for top_k in top_ks:
                file = f"{base_folder}/{model}/{dataset}/{retriever}/{top_k}/reader_output_index_0_to_6000.jsonl"
                try:
                    
                    data1 = load_jsonl(file)

                    if len(data1) in [2837, 5600, 3837]:
                        # print(f"{model} - {dataset} - {retriever} - {top_k} ---> DONE")
                        pass
                    elif len(data1)>0:
                        print(file)
                        print(f"{model} - {dataset} - {retriever} - {top_k} ---> Partial, {len(data1)}")
                    else:
                        print(file)
                        print(f"{model} - {dataset} - {retriever} - {top_k} ---> X")
                except Exception:
                    print(f"{model} - {dataset} - {retriever} - {top_k} ---> X")

                


# Check if bert score is generated 
# base_folder = READER_BASE_FOLDER
# models = "flanT5 flanUl2 llama_7b llama_70b".split(" ")
# datasets = "complete_bioasq".split(" ")
# retrievers = "bm25 colbert".split(" ")
# # top_ks = ["baseline", "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]

# for model in models:
#     for dataset in datasets:
#         for retriever in retrievers:
#             # for top_k in top_ks:
#             file = f"{base_folder}/{model}/{dataset}/{retriever}/combined_metrics.json"
#             if not os.path.exists(file):
#                 print(f"{model} - {dataset} - {retriever} ---> X")
#                 continue
            
#             data = load_json(file)
#             print(f"{model} - {dataset} - {retriever}", list(data.keys()))
#             # for k,v in data.items():
#             #     if "bertscore_f1" in v and v['bertscore_f1']>0:
#             #         continue
#             #     print(f"BERTSCORE MISSING IN {file} {k}")


#/data/user_data/jhsia2/dbqa/reader_results/llama_7b/hotpotqa/bm25/top50/ 4800




# base_folder = READER_BASE_FOLDER
# models = "flanT5 flanUl2 llama_7b llama_70b".split(" ")
# datasets = "nq".split(" ")
# retrievers = "bm25 colbert".split(" ")
# top_ks = [ "top1", "top5", "top10"]

# top1_map = {}
# top5_map = {}
# top10_map = {}
# for model in models:
#     # print(model)
#     for dataset in datasets:
        
#         for retriever in retrievers:
            
#             file = f"{base_folder}/{model}/{dataset}/{retriever}/combined_metrics.json"
#             if not os.path.exists(file):
#                 print(f"{model} - {dataset} - {retriever} ---> X")
#                 continue
#             data = load_json(file)
#             # pprint(f"{model}/{retriever}")
#             top1_map[f"{model}/{retriever}"] = round(data["top1"]["f1"]*100, 1)
#             top5_map[f"{model}/{retriever}"] = round(data["top5"]["f1"]*100, 1)
#             top10_map[f"{model}/{retriever}"] = round(data["top10"]["f1"]*100, 1)

            # print(retriever, {

            #     "top1":round(data["top1"]["f1"]*100, 1),
            #     "top5":round(data["top5"]["f1"]*100, 1),
            #     "top10":round(data["top10"]["f1"]*100, 1)
            #         })
    # print("__________________")
            
# print(" & ".join([str(x) for x in list(top10_map.values())]))
# print("top1")            
# for k,v in top1_map.items():
#     print(k," - ",v)
# print()
# print("top5")            
# for k,v in top5_map.items():
#     print(k," - ",v)
# print()
# print("top10")            
# for k,v in top10_map.items():
#     print(k," - ",v)


# retriever = "colbert"
# dataset = "nq"


# retriever_path_map = {
#         "bm25": f"{BASE_FOLDER}/retriever_results/predictions/bm25/",
#         "colbert": f"{BASE_FOLDER}/retriever_results/predictions/colbert/"
#     }
# retriever_eval_path_map = {
#         "bm25": f"{BASE_FOLDER}/retriever_results/evaluations/bm25/",
#         "colbert": f"{BASE_FOLDER}/retriever_results/evaluations/colbert/"
#     }
# dataset_map = {
#         "hotpotqa" : "hotpotqa-dev-kilt.jsonl",
#         "nq": "nq-dev-kilt.jsonl",
#         "bioasq": "bioasq.jsonl",
#         "complete_bioasq": "complete_bioasq.jsonl"
#     }

# for retriever in ["bm25", "colbert"]:
#     for dataset in ["hotpotqa"]:
#         retriever_data_path = f"{retriever_path_map[retriever]}{dataset_map[dataset]}"
#         retriever_eval_path = f"{retriever_eval_path_map[retriever]}{dataset_map[dataset]}"

#         print(retriever_data_path)
#         print(retriever_eval_path)


#         retriever_data = load_jsonl(retriever_data_path)
#         eval_data = load_jsonl(retriever_eval_path)

#         for r_info, e_info in zip(retriever_data, eval_data):
#             for par_info, par_match_info in zip(r_info["output"][0]["provenance"], e_info["doc-level results"]):
#                 par_info["wiki_par_id_match"] = par_match_info["wiki_par_id_match"]
#                 # par_info["pm_sec_id_match"] = par_match_info["wiki_par_id_match"]

#         save_jsonl(retriever_data, retriever_data_path)
                    
path = "/data/tir/projects/tir6/general/afreens/dbqa/noisy_reader_results/llama_70b/hotpotqa/colbert/top50/"
data = combine_all_files(path)
print(path)
print(len(data))

path = "/data/tir/projects/tir6/general/afreens/dbqa/noisy_reader_results/flanT5/hotpotqa/colbert/top20/"
data = combine_all_files(path)
print(path)
print(len(data))

path = "/data/tir/projects/tir6/general/afreens/dbqa/noisy_reader_results/flanT5/hotpotqa/colbert/top50/"
data = combine_all_files(path)
print(path)
print(len(data))