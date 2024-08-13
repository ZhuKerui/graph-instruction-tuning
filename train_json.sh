# Model config
MODEL_SIZE=7
# BASE_MODEL=meta-llama/Llama-2-${MODEL_SIZE}b-hf
# BASE_MODEL=mistralai/Mistral-${MODEL_SIZE}B-v0.1
BASE_MODEL=google/gemma-${MODEL_SIZE}b

# Training parameter config
export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
EPOCHS=5
MAIN_PROCESS_PORT=12303

# Data config
DATA_SOURCE=amazon
DATA_FORMAT=json
HAS_ATTRIBUTE=false
