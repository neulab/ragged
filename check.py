import os
from file_utils import BASE_FOLDER, READER_BASE_FOLDER, load_json, load_jsonl


# Check if everything is generated
base_folder = READER_BASE_FOLDER
models = "llama_70b_256_tokens".split(" ")
# datasets = "bioasq".split(" ")
datasets = "nq hotpotqa bioasq complete_bioasq".split(" ")
retrievers = "colbert".split(" ")
top_ks = ["baseline", "top1", "top2", "top3", "top5", "top10", "top20","top30", "top50"]

for model in models:
    for dataset in datasets:
        for retriever in retrievers:
            for top_k in top_ks:
                file = f"{base_folder}/{model}/{dataset}/{retriever}/{top_k}/all_data.jsonl"
                file2 = f"{base_folder}/{model}/{dataset}/{retriever}/{top_k}/reader_output_index_0_to_6000.jsonl"
                try:
                    
                    if os.path.exists(file):
                        data1 = load_jsonl(file)
                    else:
                        data1 = load_jsonl(file2)
                    if len(data1) == 0:
                        # print("earlier zero")
                        data2 = load_jsonl(file2)
                        if len(data2) != 2837 and  len(data2) != 5600:
                            print(f"{model} - {dataset} - {retriever} - {top_k} --->", len(data2), "earlier zero")
                    else:
                        if len(data1) != 2837 and  len(data1) != 5600 and len(data1) != 3837:
                            print(f"{model} - {dataset} - {retriever} - {top_k} --->", len(data1))
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