
from transformers import BertModel
from models.opt import OPTModel

def get_plm_configs(plm):
    output_size = plm.config.word_embed_proj_dim
    plm_hidden_size = plm.config.hidden_size
    plm_n_layers = plm.config.num_hidden_layers
    plm_n_heads = plm.config.num_attention_heads
    plm_n_embd = output_size // plm_n_heads
    if isinstance(plm, OPTModel):
        plm_dropout_prob = plm.config.dropout
    elif isinstance(plm, BertModel):
        plm_dropout_prob = plm.config.hidden_dropout_prob

    return output_size, plm_hidden_size, plm_n_layers, plm_n_heads, plm_n_embd, plm_dropout_prob


def freeze_plm_layers(plm, num_unfreeze_layers):
    plm_n_layers = plm.config.num_hidden_layers
    if num_unfreeze_layers < -1 or num_unfreeze_layers > plm_n_layers:
        raise ValueError(
            f"num_unfreeze_layers {num_unfreeze_layers} is not supported.")

    for param in plm.parameters():
        if num_unfreeze_layers == -1:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if num_unfreeze_layers > 0:
        if isinstance(plm, OPTModel):
            unfreeze_layers = plm.decoder.layers[-num_unfreeze_layers:]
        elif isinstance(plm, BertModel):
            unfreeze_layers = plm.encoder.layer[-num_unfreeze_layers:]
        for param in unfreeze_layers.parameters():
            param.requires_grad = True

    return plm


def get_plm_model(plm_model_name):
    if plm_model_name.startswith("facebook/opt"):
        return OPTModel.from_pretrained(plm_model_name,
                                        output_hidden_states=True)
    elif plm_model_name.startswith("bert"):
        return BertModel.from_pretrained(plm_model_name,
                                         output_hidden_states=True)
    else:
        raise ValueError(f"plm {plm_model_name} is not supported.")