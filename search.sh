OBJECTIVE="test_HR@10"

# trainer
ACCELERATOR="gpu"
MAX_EPOCHS=300
EARLY_STOPPING=10

# datamodule
TOKENIZED_LEN=30
NUM_WORKERS=4
MIN_ITEM_SEQ_LEN=5
MAX_ITEM_SEQ_LEN="None"

# prompt
USE_PROMPT="False"
PRE_SEQ_LEN=0
POST_SEQ_LEN=0
LAST_QUERY_LEN=0
PROMPT_PROJECTION="nonlinear"
PROMPT_HIDDEN_SIZE=256

# text encoder
PROJECTION_N_LAYERS=1
PROJECTION_INNER_SIZES=""

# sasrec
DIM=256
SASREC_INNER_SIZE=$[DIM*4]
SASREC_SEQ_LEN=20
SASREC_N_LAYERS=2
SASREC_N_HEADS=2
LAYER_NORM_EPS=1e-6
DROPOUT=0.5
INITIALIZER_RANGE=0.02
TOPK_LIST="5 10 20"

DATASET="MIND_large"
INPUT_TYPE="text"
POOLING_METHOD="cls"
PRETRAINED_MODEL="bert-base-uncased"
UNFREEZE=0
PLM_LR=5e-5
PLM_LR_LAYER_DECAY=1
PRECISION=16
DEVICES="1" 
 
for lr in 0.00051
do
  for bs in 64
  do
    python3 run.py --input_type $INPUT_TYPE \
                    --max_epochs $MAX_EPOCHS \
                    --early_stop_patience $EARLY_STOPPING \
                    --batch_size $bs \
                    --num_workers $NUM_WORKERS \
                    --devices $DEVICES \
                    --accelerator $ACCELERATOR \
                    --precision $PRECISION \
                    --dataset $DATASET \
                    --min_item_seq_len $MIN_ITEM_SEQ_LEN \
                    --max_item_seq_len $MAX_ITEM_SEQ_LEN \
                    --sasrec_seq_len $SASREC_SEQ_LEN \
                    --lr $lr \
                    --sasrec_hidden_size $DIM \
                    --sasrec_inner_size $SASREC_INNER_SIZE \
                    --sasrec_n_layers $SASREC_N_LAYERS \
                    --sasrec_n_heads $SASREC_N_HEADS \
                    --sasrec_layer_norm_eps $LAYER_NORM_EPS \
                    --sasrec_hidden_dropout $DROPOUT \
                    --sasrec_attention_dropout $DROPOUT \
                    --sasrec_initializer_range $INITIALIZER_RANGE \
                    --topk_list $TOPK_LIST \
                    --tokenized_len $TOKENIZED_LEN \
                    --plm_name $PRETRAINED_MODEL \
                    --plm_n_unfreeze_layers $UNFREEZE \
                    --projection_n_layers $PROJECTION_N_LAYERS \
                    --projection_inner_sizes $PROJECTION_INNER_SIZES \
                    --use_prompt $USE_PROMPT \
                    --prompt_projection $PROMPT_PROJECTION \
                    --prompt_hidden_size $PROMPT_HIDDEN_SIZE \
                    --pre_seq_len $PRE_SEQ_LEN \
                    --pooling_method $POOLING_METHOD \
                    --post_seq_len $POST_SEQ_LEN \
                    --last_query_len $LAST_QUERY_LEN \
                    --plm_lr $PLM_LR \
                    --plm_lr_layer_decay $PLM_LR_LAYER_DECAY 
  done
done
