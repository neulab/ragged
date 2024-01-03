#!/bin/bash
#SBATCH --job-name=flanUL2
#SBATCH --partition=babel-shared-long
#SBATCH --output=/home/afreens/logs/launch-tgi/flanUL2.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=200GB
#SBATCH --time=5-00:00:00

echo $SLURM_JOB_ID

model_id='google/flan-ul2'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="hf_***"

mkdir -p /scratch/afreens
text-generation-launcher --model-id $model_id --port 8101 --master-port 23457 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
# text-generation-launcher --model-id $model_id --port 9427 --master-port 23457 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
