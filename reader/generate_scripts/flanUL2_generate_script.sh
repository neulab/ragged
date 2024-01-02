#!/bin/bash
#
#SBATCH --job-name=kiltflanT5_generation
#SBATCH --output="/data/user_data/afreens/kilt/logs/flan/slurm-%A_%a.out"
#SBATCH --time=800


export PYTHONPATH=/home/afreens/ragged
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever bm25 --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9427

python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 0 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 1 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 2 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 3 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 5 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 10 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 20 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 30 --hosted_api_port 9427
python /home/afreens/ragged/reader/generate_top_k.py --model flanUl2 --retriever colbert --dataset complete_bioasq --start_offset 0 --end_offset 6000 --hosted_api_path $1 --top_k 50 --hosted_api_port 9427

