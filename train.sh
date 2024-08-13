source $1

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

MODEL_NAME=$(basename ${BASE_MODEL})
DOMAIN_DIR=../data_process/${DATA_SOURCE}/data
TEMP_DIR=../data_process/${DATA_SOURCE}/data/temp_${MODEL_NAME}
if [ ! -d $TEMP_DIR ]
then
    mkdir $TEMP_DIR
fi
cd $TEMP_DIR
FILES=`ls train_*`
cd ../../../../
if [[ -z $FILES ]];
then
    cd $DOMAIN_DIR
    FILES=`ls train_*`
    cd ../../../
    python prepare_data.py --input-dir $DOMAIN_DIR --output-dir $TEMP_DIR --files $FILES --tokenizer-model meta-llama/Llama-2-7b-hf
fi
TRAIN_FILE=${TEMP_DIR}/train_${DATA_FORMAT}.jsonl
CKPT_DIR=../data_process/${DATA_SOURCE}/ckpt/${MODEL_NAME}_qlora_${DATA_FORMAT}

cd ../src/

# Lora training
accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
	--main_process_port $MAIN_PROCESS_PORT \
    finetune.py \
    --model_name_or_path $BASE_MODEL \
    --gradient_checkpointing \
    --use_qlora \
    --use_lora \
    --use_flash_attn \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name $BASE_MODEL \
    --train_file $TRAIN_FILE \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --clip_grad_norm 0.3 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir ${CKPT_DIR} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_slow_tokenizer 

python merge_lora.py \
    --base_model_name_or_path $BASE_MODEL \
    --tokenizer_name_or_path ${CKPT_DIR}/ \
    --lora_model_name_or_path ${CKPT_DIR}/ \
    --output_dir ${CKPT_DIR}_merged/ \
    --qlora \
    --save_tokenizer
