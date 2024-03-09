
#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/analysis_framework/zeno_upload.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

conda activate py10

python "$RAGGED_PATH/analysis_framework/zeno_upload.py" --dataset $dataset \
                                                        --retriever_models 'flanUl2,flanT5,llama_70b,llama_7b'\
                                                        --reader_models 'bm25,colbert,gold'

conda deactivate
