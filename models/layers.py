import copy
import math
import torch
import torch.nn as nn
from models.utils import get_plm_configs

class PromptEncoder(torch.nn.Module):
    
    def __init__(self, plm, prompt_projection, prompt_seq_len, hidden_size,
                 prompt_hidden_size, num_hidden_layers, layer_norm_eps):
        super().__init__()
        
        _, plm_hidden_size, plm_n_layers, \
        plm_n_heads, plm_n_embd, plm_dropout_prob = get_plm_configs(plm)
        
        self.plm_n_layers = plm_n_layers
        self.plm_hidden_size = plm_hidden_size
        self.plm_n_heads = plm_n_heads
        self.plm_n_embd = plm_n_embd
        self.plm_dropout_prob = plm_dropout_prob
        self.prompt_projection = prompt_projection
        self.prompt_hidden_size = prompt_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.prompt_seq_len = prompt_seq_len
        
        self.register_buffer("tokens", torch.arange(self.prompt_seq_len).long())

        self.prompt_projection = prompt_projection
        if self.prompt_projection:
            # Use a two-layer MLP to encode the prompt
            self.embedding = torch.nn.Embedding(prompt_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prompt_hidden_size),
                torch.nn.GELU(),
                torch.nn.LayerNorm(prompt_hidden_size, layer_norm_eps),
                torch.nn.Linear(prompt_hidden_size,
                                num_hidden_layers * 2 * hidden_size))
        else:
            self.embedding = torch.nn.Embedding(
                prompt_seq_len, num_hidden_layers * 2 * hidden_size)


    def forward(self, batch_size):
        tokens = self.tokens.unsqueeze(0).expand(batch_size, -1)
        
        if self.prompt_projection:
            prefix_tokens = self.embedding(tokens)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(tokens)
        
        past_key_values = past_key_values.view(batch_size, self.prompt_seq_len,
                                               self.plm_n_layers * 2, self.plm_n_heads,
                                               self.plm_n_embd)
        
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute(2, 0, 3, 1, 4).split(2)
        
        return past_key_values


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(
            0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(
            0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(
            0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_size, inner_size, dropout_rate, layer_norm_eps):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, inner_size, kernel_size=1)
        self.relu = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.conv2 = torch.nn.Conv1d(inner_size, hidden_size, kernel_size=1)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, inputs):
        outputs = self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, L)
        outputs = self.LayerNorm(outputs + inputs)
        return outputs


class TransformerLayer(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size,
                                                       hidden_dropout_prob,
                                                       attn_dropout_prob,
                                                       layer_norm_eps)
        self.feed_forward = PointWiseFeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states,
                                                     attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        layer_norm_eps=1e-12,
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
