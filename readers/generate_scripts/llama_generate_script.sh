#!/bin/bash
#
#SBATCH --job-name=kilt_llama_generation
#SBATCH --output="/data/user_data/afreens/kilt/llama/hotpot/bm25/slurm-%A_%a.out"
#SBATCH --time=600
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 0
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 1
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 2
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 3
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 5
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 10
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 20
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 30
python /home/afreens/ragged/readers/generate_top_k.py --model llama --retriever colbert --dataset hotpotqa --start_offset $1 --end_offset $2 --hosted_api_path $3 --top_k 50
