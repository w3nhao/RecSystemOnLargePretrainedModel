OBJECTIVE="test_HR@10"

# datamodule
TOKENIZED_LEN=30
NUM_WORKERS=10
MIN_ITEM_SEQ_LEN=5
MAX_ITEM_SEQ_LEN="None"
SPLIT_TYPE="leave_one_out"

# prompt
USE_PROMPT="False"
PRE_SEQ_LEN=5
POST_SEQ_LEN=0
LAST_QUERY_LEN=0
PROMPT_PROJECTION="nonlinear"
PROMPT_HIDDEN_SIZE=256

# sasrec
SASREC_SEQ_LEN=20
SASREC_N_LAYERS=2
SASREC_N_HEADS=2
LAYER_NORM_EPS=1e-6
DROPOUT=0.1
INITIALIZER_RANGE=0.02
TOPK_LIST="5 10 20"

# text encoder
POOLING_METHOD="mean"
PRETRAINED_MODEL="facebook/opt-125m"
PLM_LR=5e-5
PLM_LR_LAYER_DECAY=0.8
PLM_WEIGHT_DECAY=0.0
PROJECTION_N_LAYERS=1
PROJECTION_INNER_SIZES=""

# pre-inference
PRE_INFERENCE_BATCH_SIZE=1
PRE_INFERENCE_DEVICES="4 5 6 7"
PRE_INFERENCE_NUM_WORKERS=0
PRE_INFERENCE_PRECISION=32
PRE_INFERENCE_LAYER_WISE="False"

# trainer
INPUT_TYPE="text"
ARCHITECTURE="sasrec"
DATASET="MIND_small"
SAMPLING_N=10000
PRE_INFERENCE="True"
UNFREEZE=0
N_NEG_SAMPLING=1
CHECK_VAL_EVERY_N_EPOCH=1
MAX_EPOCHS=150
EARLY_STOPPING=10
PRECISION=32
ACCELERATOR="gpu"
STRATEGY="none"
DEVICES="0" 
# 1e-5 7e-5 1e-4 5e-4 1e-3
# 0.0 0.01 0.1 

for dim in 512
do
  for lr in 1e-4
  do
    for wd in 0.1
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
                        --weight_decay $wd \
                        --sasrec_hidden_size $dim \
                        --sasrec_inner_size $[4*dim] \
                        --sasrec_n_layers $SASREC_N_LAYERS \
                        --sasrec_n_heads $SASREC_N_HEADS \
                        --sasrec_layer_norm_eps $LAYER_NORM_EPS \
                        --sasrec_hidden_dropout $DROPOUT \
                        --sasrec_attention_dropout $DROPOUT \
                        --sasrec_initializer_range $INITIALIZER_RANGE \
                        --topk_list $TOPK_LIST \
                        --tokenized_len $TOKENIZED_LEN \
                        --plm_name $PRETRAINED_MODEL \
                        --plm_last_n_unfreeze $UNFREEZE \
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
                        --plm_lr_layer_decay $PLM_LR_LAYER_DECAY \
                        --plm_weight_decay $PLM_WEIGHT_DECAY \
                        --strategy $STRATEGY \
                        --pre_inference $PRE_INFERENCE \
                        --pre_inference_batch_size $PRE_INFERENCE_BATCH_SIZE \
                        --pre_inference_devices $PRE_INFERENCE_DEVICES \
                        --pre_inference_precision $PRE_INFERENCE_PRECISION \
                        --pre_inference_num_workers $PRE_INFERENCE_NUM_WORKERS \
                        --pre_inference_layer_wise $PRE_INFERENCE_LAYER_WISE \
                        --split_type $SPLIT_TYPE \
                        --check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH  \
                        --architecture $ARCHITECTURE \
                        --n_neg_sampling $N_NEG_SAMPLING \
                        --sampling_n $SAMPLING_N
      done
    done
  done
done
