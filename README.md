
### how to use
Clone this repository first, and download the data from [here](https://share.weiyun.com/eJh8dB51), uncompress data.tar to the folder `data/`  
If you want to gain accessment of the data, please contact me via wenhao.deng@foxmail.com for the password.  

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

### existing program issues
if you encounter the following error:

```bash
RuntimeError: torch_shm_manager at "/opt/anaconda3/envs/plmrs/lib/python3.8/site-packages/torch/bin/torch_shm_manager": could not generate a random directory for manager socket
```
Probably the reason is that you are using a shared server, and the shared server has a limit on hard drive space. You can try to delete cache files and try again.
```bash
cd ~/.cache/huggingface/hub
rm tmp*
```
Or setting `--num_workers` to 0 and `--pre_inference_num_workers` to 0 if using pre-inference.

### thoughts of implementation
1. For super large model like OPT13B or larger, we split the model into layers and infer the embs layer by layer. It could save GPU memory when only a few layers are needed to be fine-tuned. 

2. Although we can store the pre-inferenced embs as an embedding layer inside the recommender model, it still takes GPU memory. So we store them as a numpy array and load them as a tensor in dataloader when needed. This slow down the training process, but save GPU memory. 
Take MIND_small as an example, the number of items is 52771, if we padding or truncate the item decription sequence to a fixed length 30, the size of item description matrix in float32 is 52771 * 30 * 768 * 4 Bytes = 4.9GB, which is too large to be loaded into GPU memory. 


### TODO
1. Add a new class to store the pre-inferenced embs as a numpy array.
2. BCE loss should access all the items, check [Accessing DataLoaders within LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/guides/data.html#accessing-dataloaders-within-lightningmodule).
3. When pre-inferencing, every time the script would load the tsv file of preprocessed item input_ids and attention_mask. For speeding up the pre-inference process, consider using feather/Jay format to store the item preprocessed data.


### notes
##### 1. Args of `run.py`
###### program specific args
-   `--input_type` can be `text` or `id`
-   `--dataset` can be `MIND_large` or `MIND_small`
-   `--max_epochs` is the maximum number of epochs
-   `--early_stop_patience` is the number of epochs to wait before early stopping
-   `--batch_size` is the batch size
-   `--num_workers` is the number of workers for data loading
-   `--devices` is the accelerators to use, should be specify as a list of integers: "0 1 2 3" when using multiple accelerators
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
-   `--plm_last_n_unfreeze` can be the number of layers to be unfrozen, default is 0, which means all layers are frozen. However, if you want to use all layers of the pretrained model, you should set it to -1, rather than pretrain model's `num_hidden_layers`, which would caused an ValueError in this programe.
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
-   `--post_seq_len` is the length of deep suffix prompt, only used when model is OPT
-   `--last_query_len` is the length of last shallow prompt, only used when model is OPT

###### pre-inference specific args
-   `--pre_inference` can be `True` or `False`, if it is `True`, then use `pre_inference_batch_size`, `pre_inference_devices` and `pre_inference_precision` to do inference before training using the frozen part of PLM model
-   `--pre_inference_batch_size` is the batch size of inference
-   `--pre_inference_devices` is the devices of inference
-   `--pre_inference_precision` is the precision of inference
-   `--pre_inference_num_workers` is the number of workers for data loading of inference

##### 2. manually inferencing before traininig
Use command like following:
```bash
python datamodules/preinference.py \ 
    --dataset "MIND_small" \
    --plm_name "facebook/opt-125m" \
    --sasrec_seq_len 20 \
    --tokenized_len 30 \
    --min_item_seq_len 5 \
    --max_item_seq_len None \
    --pre_inference_devices "0 1 2 3 4 5 6 7" \
    --pre_inference_precision 32 \
    --pre_inference_batch_size 1 \
    --pre_inference_num_workers 4 \
    --last_n_unfreeze 0
```

Args of `preinference.py`:
-   `--dataset` can be `MIND_large`, `MIND_small`, `hm` or `bilibili`
-   `--plm_name` can be `facebook/opt-125m` to `facebook/opt-66b` or `bert-base-uncased` to `bert-large-uncased`
-   `--sasrec_seq_len` is the length of item sequence for SASRec
-   `--tokenized_len` is the length of tokenized sequence for PLM
-   `--min_item_seq_len` is the minimum length of item sequence after preprocessing
-   `--max_item_seq_len` is the maximum length of item sequence after preprocessing
-   `--pre_inference_devices` is the devices of inference
-   `--pre_inference_precision` is the precision of inference
-   `--pre_inference_batch_size` is the batch size of inference
-   `--pre_inference_num_workers` is the number of workers for data loading of inference
-   `--plm_last_n_unfreeze` is the number of layers to be unfrozen, default is 0, which means all layers are frozen. However, if you want to use all layers of the pretrained model, you should set it to -1, rather than pretrain model's `num_hidden_layers`, which would caused an ValueError in this programe. In the pre-inference stage, the unfrozen layers are not used.
