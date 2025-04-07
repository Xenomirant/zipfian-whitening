MODEL_NAMES="bert-base-uncased FacebookAI/roberta-base microsoft/deberta-base"
POOLING_NAMES="avg_first_last avg_first_last_uniform_centering avg_first_last_uniform_whitening avg_first_last_pseudo-uniform_centering avg_first_last_pseudo-uniform_whitening"
WHITENING_MODES="pca zca cholesky"
for MODEL_NAME in $MODEL_NAMES; do
    for POOLING_NAME in $POOLING_NAMES; do
        if [[ $POOLING_NAME =~ "whitening" || $POOLING_NAME =~ "centering" ]]; then
            for WHITENING_MODE in $WHITENING_MODES; do
                echo "MODEL_NAME: $MODEL_NAME"
                echo "POOLING_NAME: $POOLING_NAME"
                echo "WHITENING_MODE: $WHITENING_MODE"
                python evaluation.py --model_name_or_path "$MODEL_NAME" --pooler "$POOLING_NAME" --whitening_mode "$WHITENING_MODE" --task_set full --mode test
            done
        else
            echo "MODEL_NAME: $MODEL_NAME"
            echo "POOLING_NAME: $POOLING_NAME"
            python evaluation.py --model_name_or_path "$MODEL_NAME" --pooler "$POOLING_NAME" --task_set full --mode test
        fi
    done
done
