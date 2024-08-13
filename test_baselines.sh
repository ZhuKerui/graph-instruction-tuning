export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
MODEL_SOURCE=baseline
MODELS=(
    # meta-llama/Llama-2-7b-chat-hf \
    # mistralai/Mistral-7B-Instruct-v0.1 \
    "google/gemma-1.1-7b-it google/gemma-1.1-7b-it" \
)

TEST_DOMAINS=(\
    maple \
    amazon \
)
TEST_TYPES=(\
    natural \
    json \
    dot \
)

BATCH_SIZE_PER_GPU=1