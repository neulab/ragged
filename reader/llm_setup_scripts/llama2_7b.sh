#!/bin/bash
#SBATCH --job-name=Llama-2-7b
#SBATCH --partition=babel-shared-long
#SBATCH --output=/home/afreens/logs/launch-tgi/Llama-2-7b.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=50GB
#SBATCH --time=1000

echo $SLURM_JOB_ID

model_id='meta-llama/Llama-2-7b-hf'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="hf_***"

mkdir -p /scratch/afreens
text-generation-launcher --model-id $model_id --port 8102 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
# text-generation-launcher --model-id $model_id --port 9426 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
