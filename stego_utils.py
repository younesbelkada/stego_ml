import torch
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union


def _init_sequence_length_for_generation(
		input_ids: torch.LongTensor, max_length: int
	) -> Tuple[torch.Tensor, torch.Tensor, int]:
		unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
		sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

		cur_len = input_ids.shape[-1]
		return sequence_lengths, unfinished_sequences, cur_len
def _update_seq_length_for_generation(
		sequence_lengths: torch.LongTensor,
		unfinished_sequences: torch.LongTensor,
		cur_len: int,
		is_eos_in_next_token: torch.BoolTensor,
	) -> Tuple[torch.LongTensor, torch.LongTensor]:
		# check if sentence is not finished yet
		is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

		# update sentence length
		sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
		unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
		return sequence_lengths, unfinished_sequences


def get_config(model):
	model_kwargs = {}
	model_kwargs['modelvocab_size'] = model.config.vocab_size
	model_kwargs['n_ctx'] = model.config.n_ctx
	model_kwargs['n_positions'] = model.config.n_positions
	model_kwargs['n_embd'] = model.config.n_embd
	model_kwargs['n_layer'] = model.config.n_layer
	model_kwargs['n_head'] = model.config.n_head
	model_kwargs['n_inner'] = model.config.n_inner
	model_kwargs['activation_function'] = model.config.activation_function
	model_kwargs['resid_pdrop'] = model.config.resid_pdrop
	model_kwargs['embd_pdrop'] = model.config.embd_pdrop
	model_kwargs['attn_pdrop'] = model.config.attn_pdrop
	model_kwargs['layer_norm_epsilon'] = model.config.layer_norm_epsilon
	model_kwargs['initializer_range'] = model.config.initializer_range
	model_kwargs['summary_type'] = model.config.summary_type
	model_kwargs['summary_use_proj'] = model.config.summary_use_proj
	model_kwargs['summary_activation'] = model.config.summary_activation
	model_kwargs['summary_first_dropout'] = model.config.summary_first_dropout
	model_kwargs['summary_proj_to_labels'] = model.config.summary_proj_to_labels
	model_kwargs['gradient_checkpointing'] = model.config.gradient_checkpointing
	model_kwargs['bos_token_id'] = model.config.bos_token_id
	model_kwargs['eos_token_id'] = model.config.eos_token_id
	model_kwargs['max_length'] = model.config.max_length
	return model_kwargs