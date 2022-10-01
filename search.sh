OBJECTIVE="test_HR@10"

MAX_EPOCHS=300
EARLY_STOPPING=10
TOKENIZED_LEN=30
NUM_WORKERS=6
MIN_ITEM_SEQ_LEN=5
MAX_ITEM_SEQ_LEN="None"

# sasrec
DIM=256
DROPOUT=0.5
SASREC_SEQ_LEN=20
NUM_BLOCKS=2
NUM_HEADS=2
LAYER_NORM_EPS=1e-6
USE_MLP_PROJECTION="no"
MLP_LAYERS_NUM=4
MLP_INNER_SIZE="3136 784 64"

# prompt
PRE_SEQ_LEN=0
POST_SEQ_LEN=0
LAST_QUERY_LEN=0
PROMPT_PROJECTION="nonlinear"
PROMPT_HIDDEN_SIZE=256

DATASET="MIND_large"
INPUT_TYPE="text"
POOLING_TYPE="mean"
PRETRAINED_MODEL="facebook/opt-1.3b"
UNFREEZE=2

DEVICES="1 2 3 4 5 6 7"

for lr in 3e-4
do
  for bs in 192
  do
    python3 run.py --lr $lr \
                    --epochs $MAX_EPOCHS \
                    --early_stop_patience $EARLY_STOPPING \
                    --batch_size $bs \
                    --input_type $INPUT_TYPE \
                    --dataset $DATASET \
                    --dim $DIM \
                    --num_workers $NUM_WORKERS \
                    --devices $DEVICES \
                    --num_blocks $NUM_BLOCKS \
                    --num_heads $NUM_HEADS \
                    --dropout $DROPOUT \
                    --unfreeze $UNFREEZE \
                    --plm $PRETRAINED_MODEL \
                    --sasrec_seq_len $SASREC_SEQ_LEN \
                    --tokenized_len $TOKENIZED_LEN \
                    --layer_norm_eps $LAYER_NORM_EPS \
                    --min_item_seq_len $MIN_ITEM_SEQ_LEN \
                    --max_item_seq_len $MAX_ITEM_SEQ_LEN \
                    --use_mlp_projection $USE_MLP_PROJECTION \
                    --mlp_layers_num $MLP_LAYERS_NUM \
                    --mlp_inner_size $MLP_INNER_SIZE  \
                    --prompt_projection $PROMPT_PROJECTION \
                    --prompt_hidden_size $PROMPT_HIDDEN_SIZE \
                    --pre_seq_len $PRE_SEQ_LEN \
                    --pooling_type $POOLING_TYPE \
                    --post_seq_len $POST_SEQ_LEN \
                    --last_query_len $LAST_QUERY_LEN
  done
done
