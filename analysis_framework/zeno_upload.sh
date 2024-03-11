
#!/bin/bash

SCRIPT_PATH="$(realpath "$0")"
RAGGED_PATH="${SCRIPT_PATH%/analysis_framework/zeno_upload.sh}"
export PYTHONPATH="$PYTHONPATH:$RAGGED_PATH"

python "$RAGGED_PATH/analysis_framework/zeno_upload.py" --dataset $dataset \
                                                        --retriever_models 'no_context,gold,flanUl2,flanT5,llama_70b,llama_7b'\
                                                        --reader_models 'bm25,colbert,gold'\
                                                        --k_list 1,5,10,20\
                                                        --zeno_api_key $zeno_api_key\
                                                        --zeno_project_name $zeno_project_name\
                                                        --create_project

