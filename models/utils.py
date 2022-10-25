from transformers import OPTModel, BertModel
import torch

PRETRAIN_MODEL_ABBR = {
    "facebook/opt-125m": "OPT125M",
    "facebook/opt-350m": "OPT350M",
    "facebook/opt-1.3b": "OPT1.3B",
    "facebook/opt-2.7b": "OPT2.7B",
    "facebook/opt-6.7b": "OPT6.7B",
    "facebook/opt-13b": "OPT13B",
    "facebook/opt-30b": "OPT30B",
    "facebook/opt-66b": "OPT66B",
    "bert-base-uncased": "BERTBASE",
    "bert-large-uncased": "BERTLARGE",
}

def gather_indexes(output, gather_index):
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)


def mean_pooling(embs, mask):
    mask_expanded = mask.unsqueeze(-1).expand_as(embs)
    sum_embs = (embs * mask_expanded).sum(1)
    num_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    output = sum_embs / num_mask
    return output


def last_pooling(embs, mask):
    batch_size = mask.shape[0]
    sentence_len = mask.shape[1]
    last_embs_idx = mask.sum(dim=1) - 1
    cumsum_idx = torch.tensor(sentence_len).expand(batch_size).cumsum(
        0).type_as(last_embs_idx)
    last_embs_idx[1:] += cumsum_idx[:-1]
    output = embs.view(-1, embs.size(-1))[last_embs_idx]
    return output


def get_plm_configs(plm):
    plm_hidden_size = plm.config.hidden_size
    plm_n_layers = plm.config.num_hidden_layers
    plm_n_heads = plm.config.num_attention_heads
    plm_n_embd = plm_hidden_size // plm_n_heads
    if isinstance(plm, OPTModel):
        output_size = plm.config.word_embed_proj_dim
        plm_dropout_prob = plm.config.dropout
    elif isinstance(plm, BertModel):
        output_size = plm.config.hidden_size
        plm_dropout_prob = plm.config.hidden_dropout_prob

    return output_size, plm_hidden_size, plm_n_layers, \
         plm_n_heads, plm_n_embd, plm_dropout_prob
