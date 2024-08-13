source $1

# Iterate over each model size
for MODEL_INFO in "${MODELS[@]}"; do
    read -a MODEL_INFO_ARR <<< "$MODEL_INFO"
    MODEL=${MODEL_INFO_ARR[0]}
    MODEL_NAME=$(basename ${MODEL})
    BASE_MODEL=${MODEL_INFO_ARR[1]}
    # Iterate over each domain
    for TEST_DOMAIN in "${TEST_DOMAINS[@]}"; do
        DOMAIN_DIR=../data_process/${TEST_DOMAIN}/data
        TEMP_DIR=../data_process/${TEST_DOMAIN}/data/temp_${MODEL_NAME}
        LOG_FILE=../data_process/${MODEL_SOURCE}/${TEST_DOMAIN}_score.txt
        cd $DOMAIN_DIR
        FILES=`ls test_*`
        cd ../../../
        python prepare_data.py --input-dir $DOMAIN_DIR --output-dir $TEMP_DIR --files $FILES --tokenizer-model meta-llama/Llama-2-7b-hf
        cd ../src
        # Iterate over each dataset
        for TEST_TYPE in "${TEST_TYPES[@]}"; do
            echo "Evaluating ${MODEL} on ${TEST_DOMAIN} ${TEST_DOMAIN} ${TEST_TYPE} using $NUM_GPUS GPUs"

            TEST_FILE=${TEMP_DIR}/test_${TEST_TYPE}.jsonl
            RESULT_FILE=../data_process/${MODEL_SOURCE}/test_results/${MODEL_NAME}/${TEST_DOMAIN}/test_${TEST_TYPE}.jsonl

            python predict.py \
                --model_name_or_path $MODEL \
                --tokenizer_name_or_path $MODEL \
                --use_slow_tokenizer \
                --input_files $TEST_FILE \
                --output_file $RESULT_FILE \
                --base_model $BASE_MODEL \
                --batch_size $BATCH_SIZE_PER_GPU &&

            echo $RESULT_FILE >> $LOG_FILE
            cd ../data_process
            python score.py --data_file $RESULT_FILE --data_source $TEST_DOMAIN >> $LOG_FILE
            cd ../src
        done
    done
done

