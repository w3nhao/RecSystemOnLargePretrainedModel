
### how to use
Clone this repository first, and download the data from [here](https://share.weiyun.com/eJh8dB51), uncompress data.tar to the folder `data/`  

The data link need a password, please contact me via wenhao.deng@foxmail.com for the password.  

The file structure should be like this:
```
├── data
│   ├── old_data
│   │   ├── MIND_small
│   │   ├── MIND_large
│   │   ├── hm
│   │   └── bilibili
│   ├── setup_scripts
│   ├── ...
├── ...
```

Then run the following command to setup the environment and the data:
```bash
conda create -n plmrs python=3.8
conda activate plmrs

wget https://mirror.sjtu.edu.cn/pytorch-wheels/cu113/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt

cd data/setup_scripts
python bilibili.py
python hm.py
python MIND_large.py
python MIND_small.py
cd ../../
```  

Then you can run the following command to train the model:
```bash
python run.py --input_type "text" --plm_name "facebook/opt-125m" --dataset "MIND_large"
```

### existing Issues
if you encounter the following error:
1. 
```bash
RuntimeError: torch_shm_manager at "/opt/anaconda3/envs/plmrs/lib/python3.8/site-packages/torch/bin/torch_shm_manager": could not generate a random directory for manager socket
```
Please try to set --num_workers to 0 and --pre_inference_num_workers to 0 if using pre-inference.

### thoughts
1. For super large model like OPT13B or larger, we split the model into layers and infer the embs layer by layer. This way could save cuda memory when only a few layers are needed to be trained. 

2. Although we can store the pre-inferenced embs as an embedding layer inside the recommender model, it still takes cuda memory. So maybe we can store them as a numpy array and load them as a tensor in dataloader when needed. This way could slow down the training process, but save cuda memory. 
Take MIND_small as an example, the number of items is 52771, if we padding or truncate the item decription sequence to a fixed length 30, the size of item description matrix in float32 is 52771 * 30 * 768 * 4 Bytes = 4.9GB, which is too large to be loaded into GPU memory. 

### TODO
1. Add a new class to store the pre-inferenced embs as a numpy array.

### Notes
##### 1. args of run.py
###### program specific args
-   `--input_type` can be `text` or `id`
-   `--dataset` can be `MIND_large` or `MIND_small`
-   `--max_epochs` is the maximum number of epochs
-   `--early_stop_patience` is the number of epochs to wait before early stopping
-   `--batch_size` is the batch size
-   `--num_workers` is the number of workers for data loading
-   `--devices` is the GPUs to use, should be specify as a list of integers: "0 1 2 3" when using multiple GPUs
-   `--accelerator` is the accelerator to use, default is `gpu`
-   `--precision` is the precision to use, default is `32`
-   `--min_item_seq_len` is the minimum length of item sequence after preprocessing
-   `--max_item_seq_len` is the maximum length of item sequence after preprocessing
-   `--strategy` is the distributed training strategy, can be `none`, `deepspeed_stage_2`, `deepspeed_stage_3`, `deepspeed_stage_2_offload`, `deepspeed_stage_3_offload`, `fsdp_offload`. if it is `none`, then use single GPU training or multi-GPU training with `ddp` accelerator 

###### sasrec specific args
-   `--sasrec_seq_len` is the length of item sequence for SASRec
-   `--weight_decay` is the weight decay for the whole model
-   `--lr` is the learning rate for SASRec
-   `--sasrec_hidden_size` is the hidden size of SASRec
-   `--sasrec_inner_size` is the inner feedforward size of SASRec
-   `--sasrec_n_layers` is the number of encoder layers of SASRec
-   `--sasrec_n_heads` is the number of heads of attention in SASRec
-   `--sasrec_layer_norm_eps` is the epsilon of layer normalization in SASRec
-   `--sasrec_hidden_dropout` is the dropout rate of hidden states in SASRec
-   `--sasrec_attention_dropout` is the dropout rate of attention weights in SASRec
-   `--sasrec_initializer_range` is the initializer range of linear layers in SASRec
-   `--topk_list` is the list of topk for evaluation metrics

###### plm specific args
-   `--tokenized_len` is the length of tokenized sequence for PLM
-   `--plm_name` can be `facebook/opt-125m` to `facebook/opt-66b` or `bert-base-uncased` to `bert-large-uncased`
-   `--plm_n_unfreeze_layers` can be the number of layers to be unfrozen, default is 0, which means all layers are frozen. However, if you want to use all layers of the pretrained model, you should set it to -1, rather than pretrain model's `num_hidden_layers`, which would caused an ValueError in this programe.
-   `--plm_lr` is the learning rate for PLM when fine-tuning
-   `--plm_lr_layer_decay` is the learning rate decay for each layer of PLM when fine-tuning
-   `--projection_n_layers` is the number of projection layers which connect PLM and SASRec
-   `--projection_inner_sizes` is the inner size of projection layers which connect PLM and SASRec, should be a list of integers and the length should be equal to `projection_n_layers` - 2, because the first and last layer are set to be PLM's hidden size and SASRec's hidden size respectively.
-   `--pooling_method` can be `mean`, `last` or `mean_last` (fusion of mean and last) for OPT model, or `mean` or `cls` for BERT model 

###### prompt specific args
-   `--use_prompt` can be `True` or `False`
-   `--prompt_projection` can be `True` or `False`
-   `--prompt_hidden_size` can be the hidden size of prompt
-   `--pre_seq_len` is the length of deep prefix prompt
-   `--post_seq_len` is the length of deep suffix prompt
-   `--last_query_len` is the length of last shallow prompt when using `post_seq_len` 

###### pre-inference specific args
-   `--pre_inference` can be `True` or `False`, if it is `True`, then use `pre_inference_batch_size`, `pre_inference_devices` and `pre_inference_precision` to do inference before training using the frozen part of PLM model
-   `--pre_inference_batch_size` is the batch size of inference
-   `--pre_inference_devices` is the devices of inference
-   `--pre_inference_precision` is the precision of inference

##### 2. If you want to manually inferencing before traininig, try command like the following:
```bash
python datamodules/preinference.py \ 
    --processed_dir data/MIND_small/MIND_small_maxlen@INF_minlen@5_toklen@30_saslen@20_processed \
    --processed_items_file news_OPT125M.processed.tsv \
    --plm_name facebook/opt-125m \
    --plm_n_unfreeze_layers 0 \
    --tokenized_len 30 \
    --batch_size 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --precision 32
```
The output of the above command is the inference result of all items descriptions text sequences, which is saved in `--processed_dir`. The output format are pytorch tensors, each tensor is the embedding of the corresponding text sequence. The length of the number of items, and the shape of each tensor is `[30, 768]` in this example. The name of the output file is `OPT125M_freeze@12_inferenced_embs_for_unfreeze@0_0.pt`. If only using one GPU, then the final number is `0`. If using multiple GPUs, then the final number is the device id of the accelerator.  

After that you should run the following command to sort the output embs by item id and to collect all the results if using multiple accelerators:
```bash
python datamodules/preinference_collect.py \
    --processed_dir data/MIND_small/MIND_small_maxlen@INF_minlen@5_toklen@30_saslen@20_processed \
    --plm_name facebook/opt-125m 
```

For datamodels/preinference.py:
-   `--processed_dir` is the directory of the processed data
-   `--processed_items_file` is the file name of the item file
-   `--plm_name` is the name of the pretrained model
-   `--plm_n_unfreeze_layers` is the number of layers to be unfrozen
-   `--tokenized_len` is the length of the tokenized text
-   `--batch_size` is the batch size
-   `--devices` is the devices to be used
-   `--precision` is the precision of the model, can be 16 or 32

For datamodels/preinference_collect.py:
-   `--processed_dir` is the directory of the processed data
-   `--plm_name` is the name of the pretrained model
