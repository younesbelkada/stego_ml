import torch
from transformers import GPT2Tokenizer, GPT2Model, set_seed, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList
from stego_utils import _init_sequence_length_for_generation, _update_seq_length_for_generation, get_config



set_seed(1)

secret_msg = "Hello I am 18"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True, pad_token_id=tokenizer.eos_token_id, max_length=30)
encoded_secret = tokenizer.encode(secret_msg, return_tensors='pt')
max_length = model.config.max_length

## pre-conditionning

input_ids = tokenizer.encode('My name is Younes', return_tensors='pt')

logits_processor = LogitsProcessorList()
sequence_lengths, unfinished_sequences, cur_len = _init_sequence_length_for_generation(input_ids, max_length)

i = 0
ranks = []
while cur_len < max_length:
	#print(cur_len, max_length)
	outputs = model(input_ids, return_dict=True)
	eos_token_id = model.config.eos_token_id
	next_token_logits = outputs.logits[:, -1, :]
	scores = logits_processor(input_ids, next_token_logits)
	index = encoded_secret[0, i]
	_, sorted_scores = torch.sort(scores)
	ranks.append(sorted_scores[0, index].item())
	next_tokens = torch.argmax(scores, dim=-1)
	#next_tokens = sorted_scores[0, index]
	# add code that transfomers next_tokens to tokens_to_add
	if eos_token_id is not None:
		assert model.config.pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
		next_tokens = next_tokens * unfinished_sequences + (model.config.pad_token_id) * (1 - unfinished_sequences)

	# add token and increase length by one
	input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

	# update sequence length
	if eos_token_id is not None:
		sequence_lengths, unfinished_sequences = _update_seq_length_for_generation(
			sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
		)

	# update model kwargs
	model_kwargs = get_config(model)
	model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)

	# stop when there is a </s> in each sentence, or if we exceed the maximul length
	if unfinished_sequences.max() == 0:
		break

	# increase cur_len
	cur_len = cur_len + 1
	i += 1
	if i >= encoded_secret.shape[1]:
		break

print("Secret message: ",secret_msg)
print("Ranks: ", ranks)

model_C = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True, pad_token_id=tokenizer.eos_token_id, max_length=300)
max_length = model_C.config.max_length

## pre-conditionning

input_ids = tokenizer.encode('In a shocking finding, scientist discovered a herd of unicorns', return_tensors='pt')

logits_processor = LogitsProcessorList()
sequence_lengths, unfinished_sequences, cur_len = _init_sequence_length_for_generation(input_ids, max_length)

i = 0
#ranks = []
print(cur_len)
while cur_len < max_length:
	#print(cur_len, max_length)
	outputs = model_C(input_ids, return_dict=True)
	eos_token_id = model_C.config.eos_token_id
	next_token_logits = outputs.logits[:, -1, :]
	scores = logits_processor(input_ids, next_token_logits)
	#index = ranks[i]
	_, sorted_scores = torch.sort(scores)
	next_tokens = torch.where(sorted_scores==ranks[i])[1][0].long()
	#exit(0)
	#ranks.append(sorted_scores[0, index].item())
	#next_tokens = torch.argmax(scores, dim=-1)
	#next_tokens = scores[0, index]
	#print(next_tokens[:, None])
	# add code that transfomers next_tokens to tokens_to_add
	if eos_token_id is not None:
		assert model_C.config.pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
		next_tokens = next_tokens * unfinished_sequences + (model_C.config.pad_token_id) * (1 - unfinished_sequences)

	# add token and increase length by one
	input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

	# update sequence length
	if eos_token_id is not None:
		sequence_lengths, unfinished_sequences = _update_seq_length_for_generation(
			sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
		)

	# update model kwargs
	model_kwargs = get_config(model_C)
	model_kwargs = model_C._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=model_C.config.is_encoder_decoder)

	# stop when there is a </s> in each sentence, or if we exceed the maximul length
	if unfinished_sequences.max() == 0:
		break

	# increase cur_len
	cur_len = cur_len + 1
	#print(tokenizer.decode(input_ids[0]))
	i += 1
	if i >= encoded_secret.shape[1]:
		break

#print(ranks)
#print(secret_msg)


shared_msg = tokenizer.decode(input_ids[0])
shared_msg_encoded = tokenizer.encode(shared_msg, return_tensors='pt')
print("shared_msg : ", shared_msg)

#input_ids = tokenizer.encode('In a shocking finding, scientist discovered a herd of unicorns', return_tensors='pt')

input_ids = tokenizer.encode('In a shocking finding, scientist discovered a herd of unicorns', return_tensors='pt')
logits_processor = LogitsProcessorList()
sequence_lengths, unfinished_sequences, cur_len = _init_sequence_length_for_generation(input_ids, max_length)

i = 0
#ranks = []
print(cur_len)
print(shared_msg_encoded[:, 12:][0][i])
ranks = []
while cur_len < max_length:
	#print(cur_len, max_length)
	outputs = model_C(input_ids, return_dict=True)
	eos_token_id = model_C.config.eos_token_id
	next_token_logits = outputs.logits[:, -1, :]
	scores = logits_processor(input_ids, next_token_logits)

	
	_, sorted_scores = torch.sort(scores)
	ranks.append(torch.where(sorted_scores==shared_msg_encoded[:, 12:][0][i])[1][0].item())
	

	next_tokens = torch.Tensor([ranks[i]]).long()
	

	#print(next_tokens[:, None])
	# add code that transfomers next_tokens to tokens_to_add
	if eos_token_id is not None:
		assert model_C.config.pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
		next_tokens = next_tokens * unfinished_sequences + (model_C.config.pad_token_id) * (1 - unfinished_sequences)

	# add token and increase length by one
	input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

	# update sequence length
	if eos_token_id is not None:
		sequence_lengths, unfinished_sequences = _update_seq_length_for_generation(
			sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
		)

	# update model kwargs
	model_kwargs = get_config(model_C)
	model_kwargs = model_C._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=model_C.config.is_encoder_decoder)

	# stop when there is a </s> in each sentence, or if we exceed the maximul length
	if unfinished_sequences.max() == 0:
		break

	# increase cur_len
	cur_len = cur_len + 1
	#print(tokenizer.decode(input_ids[0]))
	i += 1
	if i >= shared_msg_encoded[:, 12:].shape[1]:
		break



#exit(0)

#print(tokenizer.decode(encoded_secret[0], skip_special_tokens=True))

print(ranks)
"""text = "Hello I am Younes"
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
print(output)"""