from transformers.models.opt.modeling_opt import (
    OPTModel, 
    OPTPreTrainedModel,
    OPTDecoderLayer,
    OPTLearnedPositionalEmbedding,
    _make_causal_mask,
    _expand_mask,
    OPT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    OPT_START_DOCSTRING,
    )
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

import random
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.opt.configuration_opt import OPTConfig


logger = logging.get_logger(__name__)


def check_keep_decoders_range(
    config: OPTConfig,
    keep_embed_layer: bool,
    keep_decoders_range: Optional[Tuple[int]]):
    # the keep_decoders_range tuple is a left-closed and right-closed interval [a, b] as input 
    if keep_decoders_range is None:
        return None
    else:
        assert len(keep_decoders_range) == 2
    
    range_start = keep_decoders_range[0]
    range_end = keep_decoders_range[1]
    assert isinstance(range_start, int) and isinstance(range_end, int)
    
    if range_start != 0 and keep_embed_layer:
        raise ValueError(
            "You can't keep keep the embedding layer since you drop the first decoder layer"
        )
        
    if range_start < 0:
        range_start = config.num_hidden_layers + range_start
    
    range_end += 1
    if range_end <= 0:
        range_end = config.num_hidden_layers + range_end
    
    assert range_end <= config.num_hidden_layers
    assert range_start <= range_end
    
    # the keep_decoders_range list is a left-closed and right-open interval [a, b) as output 
    return (range_start, range_end)


class PartialOPTDecoder(OPTPreTrainedModel):
    
    def __init__(
        self, 
        config: OPTConfig,
        keep_embed_layer: bool = True,
        keep_decoders_range: Optional[Tuple[int]] = (0, -1)):
        r"""
        Args:
            config (:obj:`OPTConfig`): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the configuration.
                Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
            keep_embed_layer (:obj:`bool`): Whether to keep the embedding layer.
            keep_decoders_range (:obj:`Tuple[int]`): A left-closed and right-closed interval as input [a, b], the range of decoders to keep.
        """
        super().__init__(config)
        keep_decoders_range = check_keep_decoders_range(config, keep_embed_layer, keep_decoders_range)
        self.keep_embed_layer = keep_embed_layer
        self.keep_decoders_range = keep_decoders_range
        
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        
        if not self.keep_embed_layer and self.keep_decoders_range is None:
            raise ValueError("You can't drop both the embedding layer and all the decoders.")
        
        if self.keep_embed_layer:
            self.padding_idx = config.pad_token_id
            self.max_target_positions = config.max_position_embeddings
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
            self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

            if config.word_embed_proj_dim != config.hidden_size:
                self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
            else:
                self.project_in = None
                
        if self.keep_decoders_range is not None:
            if self.keep_decoders_range[1] == config.num_hidden_layers:
            # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
            # with checkpoints that have been fine-tuned before transformers v4.20.1
            # see https://github.com/facebookresearch/metaseq/pull/164
                if config.do_layer_norm_before and not config._remove_final_layer_norm:
                    self.final_layer_norm = nn.LayerNorm(config.hidden_size)
                else:
                    self.final_layer_norm = None

            self.layers = nn.ModuleList()
            for i in range(self.keep_decoders_range[0], self.keep_decoders_range[1]):
                self.layers.add_module(str(i), OPTDecoderLayer(config))
            
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if self.keep_embed_layer:
            return self.embed_tokens
        else:
            raise NotImplementedError("The model drops the input embeddings layer.")

    def set_input_embeddings(self, value):
        if self.keep_embed_layer:
            self.embed_tokens = value
        else:
            raise NotImplementedError("The model drops the input embeddings layer.")

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.keep_embed_layer:
            assert input_ids is not None, "You have to specify input_ids."
        else:
            assert inputs_embeds is not None, "You have to specify inputs_embeds"

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class PartialOPTModel(OPTModel):
    
    def __init__(self,
                 config: OPTConfig,
                 keep_embed_layer: bool = True,
                 keep_decoders_range: Optional[Tuple[int]] = (0, -1)):

        r"""
        Args:
            config (:obj:`OPTConfig`): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the configuration.
                Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
            keep_embed_layer (:obj:`bool`): Whether to keep the embedding layer.
            keep_decoders_range (:obj:`Tuple[int]`): A left-closed and right-closed interval as input [a, b], the range of decoders to keep.
        """
        super(OPTModel, self).__init__(config)
        self.decoder = PartialOPTDecoder(config, keep_embed_layer, keep_decoders_range)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )