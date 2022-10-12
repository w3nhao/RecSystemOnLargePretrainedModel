import torch
import torch.nn as nn
from models.layers import TransformerEncoder


class SASRec(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    """

    def __init__(
        self,
        lr,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout,
        attention_dropout,
        layer_norm_eps,
        initializer_range,
        seq_len,
    ):
        super(SASRec, self).__init__()

        # load parameters info
        self.lr = lr
        self.n_layers = n_layers
        self.n_heads = n_heads
        # hidden size is the same as embedding_size
        self.hidden_size = hidden_size
        # the inner dimensionality in feed-forward layer
        self.inner_size = inner_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        # define layers and loss
        self.position_embedding = nn.Embedding(seq_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout,
            attn_dropout_prob=self.attention_dropout,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size,
                                      eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout)
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version
            # which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    def _get_attention_mask(self, attn_mask, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        extended_attention_mask = attn_mask.unsqueeze(1).unsqueeze(
            2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand(
                    (-1, -1, attn_mask.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0.0,
                                              -10000.0)
        return extended_attention_mask

    def forward(self, item_emb, attn_mask):
        position_ids = torch.arange(attn_mask.size(1),
                                    dtype=torch.long,
                                    device=attn_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(attn_mask)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self._get_attention_mask(attn_mask)

        trm_output = self.trm_encoder(input_emb,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        output = trm_output[-1]  # (batch_size, seq_len, hidden_size)
        return output
