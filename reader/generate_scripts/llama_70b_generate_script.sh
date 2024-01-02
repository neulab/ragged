#!/bin/bash
#
#SBATCH --job-name=kilt_llama_70b_generation
#SBATCH --output="/data/user_data/afreens/kilt/logs/llama/slurm-%A_%a.out"
#SBATCH --time=600

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9429

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 8103 --max_new_tokens 10
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9429 --max_new_tokens 10

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9429
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9429

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 7103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 7103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 7103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset nq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 7103 --max_new_tokens 256

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 8103 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 8103 --max_new_tokens 256
python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9103 --max_new_tokens 256
python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9103 --max_new_tokens 256
python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9103 --max_new_tokens 256
python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset hotpotqa --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9103 --max_new_tokens 256

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9429 --max_new_tokens 256

# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9429 --max_new_tokens 256
# python /home/afreens/ragged/reader/generate_top_k.py --model llama_70b_256_tokens --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9429 --max_new_tokens 256

