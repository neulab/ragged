#!/bin/bash
#SBATCH --job-name=Llama-2-70b-hf
#SBATCH --partition=babel-shared-long
#SBATCH --output=/home/jhsia2/ragged/reader/llm_setup_scripts/Llama-2-70b-hf_v2.out
#SBATCH --error=/home/jhsia2/ragged/reader/llm_setup_scripts/Llama-2-70b-hf_v2.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

export PYTHONPATH=$PYTHONPATH:/home/jhsia2/ragged

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

model_id='meta-llama/Llama-2-70b-hf'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="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"


export user=jhsia2
mkdir -p /scratch/$user
# v3 text-generation-launcher --model-id $model_id --port 7103 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
# v4 text-generation-launcher --model-id $model_id --port 9103 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
text-generation-launcher --model-id $model_id --port 8103 --master-port 23456 --shard-uds-path /scratch/$user/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
# text-generation-launcher --model-id $model_id --port 9429 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-udscd ../top  -socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
