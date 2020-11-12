from transformers import GPT2Tokenizer, GPT2Model, set_seed

set_seed(1)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2Model.from_pretrained('gpt2', return_dict=True)

text = "Hello I am Younes"
encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)
print(output.last_hidden_state)