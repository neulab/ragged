#!/bin/bash
#
#SBATCH --job-name=kilt_llama_7b_generation
#SBATCH --output="/data/user_data/afreens/kilt/logs/llama/slurm-%A_%a.out"
#SBATCH --time=600
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9426

python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9426
python /home/afreens/ragged/reader/generate_top_k.py --model llama_7b --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9426
