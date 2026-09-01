import os
from importlib.metadata import version
import tiktoken

print("tiktoken version: ",version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

with open(os.path.join(os.path.dirname(__file__), '..', 'the_verdict.txt'), "r") as file:
    raw_text = file.read()
file.close()

enc_text = tokenizer.encode(raw_text)

print("Length of encoded text: ", len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
for i in range(1,context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

#print(tokenizer.decode(enc_text))


