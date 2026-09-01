import os
import os
import re 
from simple_tokenizer import SimpleTokenizerV2

with open(os.path.join(os.path.dirname(__file__), '..', 'the_verdict.txt'), "r") as file:
    raw_text = file.read()
file.close()

raw_text = raw_text.replace("\n", " ")

preprocessed = re.split(r'([,.:;?_!"\'()]|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
vocab_size = len(vocab.items())


tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the Summit terraces of the place"
text = "<|endoftext|>".join((text1,text2))


ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

