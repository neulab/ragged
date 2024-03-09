export PYTHONPATH=$PYTHONPATH:$ragged_dir:$pyserini_dir

conda activate kilt

python ${kilt_dir}scripts/execute_retrieval.py -m bm25 -o ${prediction_dir}/bm25 --test_config KILT/kilt/configs/${dataset}.json

conda deactivate

conda activate py10