#!/bin/bash
#SBATCH --job-name=Llama7b
#SBATCH --partition=general
#SBATCH --output=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/llama_7b.out
#SBATCH --error=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/llama_7b.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=50GB
#SBATCH --time=12:00:00

source /home/jhsia2/.bashrc
conda activate tgi-env

echo $SLURM_JOB_ID

cd /home/jhsia2/text-generation-inference

OPENSSL_DIR=`pwd`/.openssl \
OPENSSL_LIB_DIR=`pwd`/.openssl/lib \
OPENSSL_INCLUDE_DIR=`pwd`/.openssl/include \
BUILD_EXTENSIONS=false \
    make install-server


model_id='meta-llama/Llama-2-7b-hf'

echo $model_id

export HUGGING_FACE_HUB_TOKEN="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"

export user=jhsia2
mkdir -p /scratch/$user
text-generation-launcher --model-id $model_id --port 8103 --master-port 23456 --shard-uds-path /scratch/$user/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096
# text-generation-launcher --model-id $model_id --port 9426 --master-port 23456 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 4000 --max-total-tokens 4096

# 4-23 8102
# 4-23 8104
# 0-37 8105

