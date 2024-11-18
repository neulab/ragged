#!/bin/bash

# Define combinations of parameters
rerankers=("bge_large" "bge_v2_m3")
retrievers=("colbert" "bm25")
ks=(50 100)

# rerankers=("bge_large")
# retrievers=("colbert" "bm25")
# ks=(100)


# Loop through all combinations
for reranker in "${rerankers[@]}"; do
  for retriever in "${retrievers[@]}"; do
    for k in "${ks[@]}"; do
      # Generate a job name
      job_name="${retriever}_${reranker}_nq_${k}"
      
      # Submit the job
      sbatch --job-name="$job_name" \
             --output="${job_name}.out" \
             --error="${job_name}.err" \
             --export=ALL,reranker="$reranker",retriever="$retriever",k="$k" \
             reranker.sh
    done
  done
done
