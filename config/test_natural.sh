export CUDA_VISIBLE_DEVICES=3
NUM_GPUS=1
DATA_FORMAT=natural
MODEL_SOURCE=maple
MODELS=( \
    # "../data_process/${MODEL_SOURCE}/ckpt/Llama-2-7b-hf_qlora_${DATA_FORMAT}_merged" \
    # "../data_process/${MODEL_SOURCE}/ckpt/Mistral-7B-v0.1_qlora_${DATA_FORMAT}_merged" \
    "../data_process/${MODEL_SOURCE}/ckpt/gemma-7b_qlora_${DATA_FORMAT}_merged google/gemma-7b" \
    "../data_process/${MODEL_SOURCE}/ckpt/gemma-2b_qlora_${DATA_FORMAT}_merged google/gemma-2b" \
)

TEST_DOMAINS=(\
    amazon \
    maple \
)
TEST_TYPES=(\
"${DATA_FORMAT}" \
)

BATCH_SIZE_PER_GPU=1
