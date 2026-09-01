from dataset import GPTDatasetV1, create_dataloader_v1
import torch
from torch.utils.data import Dataset, DataLoader


with open("the_verdict.txt","r") as file:
    raw_text = file.read()
file.close()

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text,batch_size = 8, max_length = max_length,
    stride = max_length,shuffle = False
)
data_iter = iter(dataloader)
inputs, target = next(data_iter)

token_embeddings = token_embedding_layer(inputs)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

