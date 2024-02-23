#!/bin/bash
#SBATCH --job-name=4flanT5
#SBATCH --partition=general
#SBATCH --output=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/4flanT5.out
#SBATCH --error=/home/jhsia2/ragged/reader/llm_setup_scripts/logs/4flanT5.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --mem=100GB
#SBATCH --time=800

source /home/jhsia2/.bashrc
conda activate tgi-env

OPENSSL_DIR=`pwd`/.openssl \
OPENSSL_LIB_DIR=`pwd`/.openssl/lib \
OPENSSL_INCLUDE_DIR=`pwd`/.openssl/include \
BUILD_EXTENSIONS=false \
    make install-server

    
echo $SLURM_JOB_ID

model_id='google/flan-t5-xxl'

echo $model_id

export user=jhsia2

export HUGGING_FACE_HUB_TOKEN="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"

mkdir -p /scratch/$user
text-generation-launcher --model-id $model_id --port 5300 --master-port 23457 --shard-uds-path /scratch/$user/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
# text-generation-launcher --model-id $model_id --port 9428 --master-port 23457 --shard-uds-path /scratch/afreens/tgi-uds-socket-1 --huggingface-hub-cache /data/tir/projects/tir2/models/tgi_cache/hub --max-input-length 2000 --max-total-tokens 2048
